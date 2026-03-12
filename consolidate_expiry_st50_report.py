#!/usr/bin/env python3
"""
Consolidate weekly/monthly trades and PnL from:
- backtesting_expiry (Tuesday expiry days) -> DYNAMIC_ATM from analysis_output_latest.csv
- backtesting (remaining 4 days per week) -> DYNAMIC_OTM from analysis_output_latest.csv

Date assignment:
- If date is in backtesting_expiry BACKTESTING_DAYS -> use expiry data (DYNAMIC_ATM).
- Otherwise (date in backtesting BACKTESTING_DAYS) -> use st50 data (DYNAMIC_OTM).

Data source: analysis_output_latest.csv in each project (not HTML).
Weekly grouping: by expiry week (Wednesday to Tuesday), keyed by week-ending Tuesday date.
Missing or filtered days: PnL=0, trades=0.

Output: consolidated_entry2_weekly.csv only. Each row = one expiry week; month_label assigns
the 4 weekly expiries to the month (e.g. FEB 2026 = 03-02, 10-02, 17-02, 24-02).
"""

import csv
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Paths (project root = script dir)
PROJECT_ROOT = Path(__file__).resolve().parent
EXPIRY_CONFIG = PROJECT_ROOT / "backtesting_expiry" / "backtesting_config.yaml"
ST50_CONFIG = PROJECT_ROOT / "backtesting" / "backtesting_config.yaml"
EXPIRY_CSV = PROJECT_ROOT / "backtesting_expiry" / "data" / "analysis_output" / "consolidated" / "analysis_output_latest.csv"
ST50_CSV = PROJECT_ROOT / "backtesting" / "data_st50" / "analysis_output" / "consolidated" / "analysis_output_latest.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
WEEKLY_CSV = OUTPUT_DIR / "consolidated_entry2_weekly.csv"
WEEKLY_PDF = OUTPUT_DIR / "fact_sheet_weekly_expiry_report.pdf"


def load_backtesting_days(config_path: Path, key_path: tuple = ("BACKTESTING_EXPIRY", "BACKTESTING_DAYS")) -> list:
    """Load BACKTESTING_DAYS from config."""
    if not config_path.exists():
        return []
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    c = cfg
    for k in key_path:
        c = c.get(k) if isinstance(c, dict) else None
        if c is None:
            return []
    return list(c) if isinstance(c, (list, tuple)) else []


def date_str_to_day_label(date_str: str) -> str:
    """'2026-02-03' -> 'FEB03'."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        return f"{d.strftime('%b').upper()}{d.strftime('%d')}"
    except Exception:
        return ""


def week_ending_tuesday(date_str: str) -> str:
    """
    Return the week-ending Tuesday (YYYY-MM-DD) for the given date.
    Week runs Wed->Tue, so for any date we get the next Tuesday (or same day if Tue).
    """
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    # weekday: Mon=0, Tue=1, ..., Sun=6. Days until next Tue: (1 - weekday) % 7
    days_ahead = (1 - d.weekday()) % 7
    tuesday = d + timedelta(days=days_ahead)
    return tuesday.strftime("%Y-%m-%d")


def load_csv_pnl_trades(csv_path: Path, strike_type: str) -> dict:
    """
    Read analysis_output_latest.csv; filter by Strike Type; return dict day_label -> (pnl_float, trades_int, winning_trades_int).
    Uses Filtered P&L, Filtered Trades (fallback Total Trades), Winning Trades.
    """
    out = {}
    if not csv_path.exists():
        return out
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Strike Type") != strike_type:
                continue
            day_label = (row.get("day_label") or "").strip().upper()
            if not day_label:
                continue
            pnl_str = (row.get("Filtered P&L") or row.get("Un-Filtered P&L") or "0").replace("%", "").strip()
            try:
                pnl = float(pnl_str)
            except ValueError:
                pnl = 0.0
            trades_str = (row.get("Filtered Trades") or row.get("Total Trades") or "0").strip()
            try:
                trades = int(float(trades_str))
            except ValueError:
                trades = 0
            win_str = (row.get("Winning Trades") or "0").strip()
            try:
                winning_trades = int(float(win_str))
            except ValueError:
                winning_trades = 0
            out[day_label] = (pnl, trades, winning_trades)
    return out


def build_date_to_source(expiry_dates: list, st50_dates: list) -> dict:
    """date_str -> 'expiry' | 'st50'. Expiry days use expiry; all others in st50 list use st50."""
    out = {}
    for d in expiry_dates:
        out[d] = "expiry"
    for d in st50_dates:
        if d not in out:
            out[d] = "st50"
    return out


def _month_display_label(month_label: str) -> str:
    """'2026-02' -> 'February 2026'."""
    try:
        dt = datetime.strptime(month_label + "-01", "%Y-%m-%d")
        return dt.strftime("%B %Y")
    except Exception:
        return month_label


# Slippage/STT assumption for report (TBD)
SLIPPAGE_STT_PCT = 0.5


def build_weekly_expiry_pdf(
    weekly_rows: list,
    total_pnl: float,
    total_trades: int,
    win_rate_pct: float,
    output_path: Path,
) -> None:
    """Generate a 1-page professional PDF: month-divided weekly expiry report."""
    if not REPORTLAB_AVAILABLE:
        print("reportlab not installed; skipping PDF generation.")
        return
    if not weekly_rows:
        print("No weekly data; skipping PDF.")
        return

    # Professional color palette
    header_bg = colors.HexColor("#1e3a5f")
    header_fg = colors.white
    row_alt = colors.HexColor("#f0f4f8")
    subtotal_bg = colors.HexColor("#d4e4f4")
    summary_bg = colors.HexColor("#1e3a5f")
    summary_fg = colors.white
    grid_color = colors.HexColor("#b0bec5")

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=36,
        bottomMargin=32,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name="ReportTitle",
        parent=styles["Heading1"],
        fontSize=16,
        spaceAfter=2,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1e3a5f"),
    )
    subtitle_style = ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#546e7a"),
        spaceAfter=12,
    )
    month_style = ParagraphStyle(
        name="MonthHeader",
        parent=styles["Heading2"],
        fontSize=10,
        spaceBefore=8,
        spaceAfter=4,
        textColor=colors.HexColor("#263238"),
    )

    flow = []
    flow.append(Paragraph("Fact Sheet — Weekly Expiry Report", title_style))
    flow.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d-%b-%Y %H:%M')}  |  ATM (expiry) + OTM (st50) consolidated",
        subtitle_style,
    ))
    flow.append(Spacer(1, 4))

    by_month = defaultdict(list)
    for row in weekly_rows:
        by_month[row["month_label"]].append(row)

    font_size = 9
    row_height = 12
    col_widths = [80, 80, 56, 48]

    for month_label in sorted(by_month.keys()):
        rows_in_month = by_month[month_label]
        flow.append(Paragraph(_month_display_label(month_label), month_style))
        data = [["Expiry (Tue)", "P&L %", "Trades", "Days"]]
        for r in rows_in_month:
            tue = r["week_ending_tuesday"]
            try:
                dt = datetime.strptime(tue, "%Y-%m-%d")
                tue_short = dt.strftime("%d %b")
            except Exception:
                tue_short = tue
            pnl = r["total_pnl_pct"]
            pnl_str = f"+{pnl:.2f}%" if pnl >= 0 else f"{pnl:.2f}%"
            days_str = (r.get("days") or "").strip()
            days_count = len([x for x in days_str.split(";") if x.strip()]) if days_str else 0
            data.append([tue_short, pnl_str, str(r["total_trades"]), str(days_count)])
        month_pnl = sum(r["total_pnl_pct"] for r in rows_in_month)
        month_tr = sum(r["total_trades"] for r in rows_in_month)
        pnl_str = f"+{month_pnl:.2f}%" if month_pnl >= 0 else f"{month_pnl:.2f}%"
        data.append(["Subtotal", pnl_str, str(month_tr), ""])
        n_rows = len(data)
        t = Table(data, colWidths=col_widths, rowHeights=[row_height] * n_rows)
        st = TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("BACKGROUND", (0, 0), (-1, 0), header_bg),
            ("TEXTCOLOR", (0, 0), (-1, 0), header_fg),
            ("LINEBELOW", (0, 0), (-1, 0), 1, header_fg),
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ("ALIGN", (2, 0), (2, -1), "RIGHT"),
            ("ALIGN", (3, 0), (3, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOX", (0, 0), (-1, -1), 0.5, grid_color),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, grid_color),
        ])
        for i in range(1, n_rows - 1):
            if i % 2 == 1:
                st.add("BACKGROUND", (0, i), (-1, i), row_alt)
        st.add("BACKGROUND", (0, n_rows - 1), (-1, n_rows - 1), subtotal_bg)
        st.add("FONTNAME", (0, n_rows - 1), (-1, n_rows - 1), "Helvetica-Bold")
        st.add("LINEABOVE", (0, n_rows - 1), (-1, n_rows - 1), 0.5, colors.HexColor("#546e7a"))
        t.setStyle(st)
        flow.append(t)
        flow.append(Spacer(1, 4))

    flow.append(Spacer(1, 8))
    pnl_per_trade = (total_pnl / total_trades) if total_trades else 0.0
    grand_pnl_str = f"+{total_pnl:.2f}%" if total_pnl >= 0 else f"{total_pnl:.2f}%"
    pnl_per_trade_str = f"+{pnl_per_trade:.2f}%" if pnl_per_trade >= 0 else f"{pnl_per_trade:.2f}%"

    # Fact-sheet table: same total width as expiry tables (col_widths = 264)
    summary_table_width = sum(col_widths)
    summary_col1 = int(summary_table_width * 0.58)
    summary_col2 = summary_table_width - summary_col1
    summary_data = [
        ["Total PnL (Simple)", grand_pnl_str],
        ["Total Trades", str(total_trades)],
        ["PnL / Trade", pnl_per_trade_str],
        ["(Slippage/STT)/Trade (To Be Calculated)", f"{SLIPPAGE_STT_PCT}%"],
        ["Win Rate", f"{win_rate_pct:.1f}%"],
    ]
    t_summary = Table(
        summary_data,
        colWidths=[summary_col1, summary_col2],
        rowHeights=[row_height] * len(summary_data),
    )
    t_summary.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("BACKGROUND", (0, 0), (-1, -1), summary_bg),
        ("TEXTCOLOR", (0, 0), (-1, -1), summary_fg),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 0.5, grid_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, grid_color),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    flow.append(t_summary)

    doc.build(flow)
    print(f"Written: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load configs (backtesting uses BACKTESTING_DAYS_ST50 / BACKTESTING_DAYS_ST100)
    expiry_dates = load_backtesting_days(EXPIRY_CONFIG)
    st50_dates = load_backtesting_days(ST50_CONFIG, ("BACKTESTING_EXPIRY", "BACKTESTING_DAYS_ST50"))
    if not st50_dates:
        st50_dates = load_backtesting_days(ST50_CONFIG)  # fallback to BACKTESTING_DAYS if present
    date_to_source = build_date_to_source(expiry_dates, st50_dates)
    all_dates = sorted(date_to_source.keys())

    # Load data from CSVs (not HTML)
    expiry_data = load_csv_pnl_trades(EXPIRY_CSV, "DYNAMIC_ATM")
    st50_data = load_csv_pnl_trades(ST50_CSV, "DYNAMIC_OTM")

    # Build per-date rows (in memory only, for weekly aggregation)
    rows = []
    for date_str in all_dates:
        day_label = date_str_to_day_label(date_str)
        source = date_to_source.get(date_str, "st50")
        data = expiry_data if source == "expiry" else st50_data
        pnl, trades, winning_trades = data.get(day_label, (0.0, 0, 0))
        strike_type = "DYNAMIC_ATM" if source == "expiry" else "DYNAMIC_OTM"
        rows.append({
            "date": date_str,
            "source": source,
            "strike_type": strike_type,
            "day_label": day_label,
            "pnl_pct": round(pnl, 2),
            "trades": trades,
            "winning_trades": winning_trades,
        })

    # Weekly aggregation: by expiry week (Wed->Tue), key = week_ending_tuesday date. Assign month from that Tuesday.
    weekly = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "dates": []})
    for r in rows:
        key = week_ending_tuesday(r["date"])
        weekly[key]["pnl"] += r["pnl_pct"]
        weekly[key]["trades"] += r["trades"]
        weekly[key]["dates"].append(r["date"])

    weekly_rows = []
    for we_tue in sorted(weekly.keys()):
        v = weekly[we_tue]
        # Month from week-ending Tuesday (e.g. 2026-02-17 -> 2026-02)
        dt = datetime.strptime(we_tue, "%Y-%m-%d")
        month_label = f"{dt.year}-{dt.month:02d}"
        weekly_rows.append({
            "month_label": month_label,
            "week_ending_tuesday": we_tue,
            "total_pnl_pct": round(v["pnl"], 2),
            "total_trades": v["trades"],
            "days": ";".join(sorted(v["dates"])),
        })
    with open(WEEKLY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["month_label", "week_ending_tuesday", "total_pnl_pct", "total_trades", "days"])
        w.writeheader()
        w.writerows(weekly_rows)
    print(f"Written: {WEEKLY_CSV} ({len(weekly_rows)} weeks)")

    total_pnl = sum(r["pnl_pct"] for r in rows)
    total_trades = sum(r["trades"] for r in rows)
    total_winning_trades = sum(r["winning_trades"] for r in rows)
    win_rate_pct = (100.0 * total_winning_trades / total_trades) if total_trades else 0.0
    build_weekly_expiry_pdf(
        weekly_rows, total_pnl, total_trades, win_rate_pct, WEEKLY_PDF,
    )

    # Console summary
    print(f"\nConsolidated totals: PnL = {total_pnl:.2f}%, Trades = {total_trades}")
    print(f"Expiry days (ATM): {sum(1 for r in rows if r['source'] == 'expiry')}")
    print(f"St50 days (OTM):  {sum(1 for r in rows if r['source'] == 'st50')}")


if __name__ == "__main__":
    main()
