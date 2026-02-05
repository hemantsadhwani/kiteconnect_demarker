# access_token.py (Fully Automated, Self-Healing Module)
import os
import sys
import time
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
from kiteconnect import KiteConnect
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.common.exceptions import WebDriverException, TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from pyotp import TOTP
import platform

# Fix Windows encoding issues (charmap can't encode Unicode emoji)
# Set UTF-8 encoding for stdout/stderr on Windows
if sys.platform == 'win32':
    try:
        # Try to set UTF-8 encoding for stdout/stderr
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')
        if sys.stderr.encoding != 'utf-8':
            sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        # Python < 3.7 or encoding not available - use ASCII-safe print wrapper
        def safe_print(*args, **kwargs):
            """Print function that handles Unicode encoding errors gracefully"""
            try:
                print(*args, **kwargs)
            except UnicodeEncodeError:
                # Replace emoji with ASCII equivalents
                safe_args = []
                for arg in args:
                    if isinstance(arg, str):
                        arg = arg.replace('✅', '[OK]').replace('⚠️', '[WARN]').replace('❌', '[ERROR]')
                    safe_args.append(arg)
                print(*safe_args, **kwargs)
        
        # Replace print with safe_print for this module
        import builtins
        builtins.print = safe_print


# Token cache to reduce API rate limiting
_token_cache = {
    'access_token': None,
    'kite_client': None,
    'api_key': None,
    'last_validated': None,
    'expires_at': None,
    'user_id': None
}

def _get_cache_path(script_dir):
    """Get path to token cache file"""
    return os.path.join(script_dir, "key_secrets", "token_cache.json")

def _load_token_cache(script_dir):
    """Load token cache from file"""
    cache_path = _get_cache_path(script_dir)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
                # Convert string timestamps back to datetime objects for comparison
                if cache_data.get('last_validated'):
                    cache_data['last_validated'] = datetime.fromisoformat(cache_data['last_validated'])
                if cache_data.get('expires_at'):
                    cache_data['expires_at'] = datetime.fromisoformat(cache_data['expires_at'])
                return cache_data
        except Exception as e:
            print(f"Warning: Could not load token cache: {e}")
    return None

def _save_token_cache(script_dir, access_token, api_key, user_id=None):
    """Save token cache to file"""
    cache_path = _get_cache_path(script_dir)
    try:
        # Kite tokens typically expire at end of trading day (3:30 PM IST = 10:00 AM UTC)
        # For safety, set expiration to 6 hours from now (tokens usually last until end of day)
        now = datetime.now()
        # If it's before 3:30 PM IST (10:00 AM UTC), token expires at 3:30 PM IST today
        # Otherwise, it expires at 3:30 PM IST tomorrow
        # For simplicity, we'll cache for 6 hours (conservative)
        expires_at = now + timedelta(hours=6)
        
        cache_data = {
            'access_token': access_token,
            'api_key': api_key,
            'user_id': user_id,
            'last_validated': now.isoformat(),
            'expires_at': expires_at.isoformat()
        }
        
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save token cache: {e}")

def _is_token_cache_valid(cache_data):
    """Check if cached token is still valid"""
    if not cache_data:
        return False
    
    # Check if cache has expired
    if cache_data.get('expires_at'):
        expires_at = cache_data['expires_at']
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)
        if datetime.now() >= expires_at:
            return False
    
    # Check if last validation was too long ago (re-validate every 30 minutes for safety)
    if cache_data.get('last_validated'):
        last_validated = cache_data['last_validated']
        if isinstance(last_validated, str):
            last_validated = datetime.fromisoformat(last_validated)
        if datetime.now() - last_validated > timedelta(minutes=30):
            return False
    
    return True

def _load_key_secret_file(api_key_path: str):
    """
    Load credentials from key_secrets/api_key.txt robustly with validation.

    Expected order (whitespace or newline separated):
      0: api_key
      1: api_secret
      2: username
      3: password
      4: totp_secret
    """
    raw = open(api_key_path, 'r').read().strip()
    # Support either space or newline separated values
    parts = [p for p in raw.replace('\r', '\n').replace('\t', ' ').replace(',', ' ').split() if p]
    if len(parts) < 5:
        raise ValueError(
            f"key_secrets/api_key.txt is missing fields. Found {len(parts)} values; expected 5 (api_key, api_secret, username, password, totp_secret)."
        )
    api_key, api_secret, username, password, totp_secret = parts[:5]
    return api_key, api_secret, username, password, totp_secret


def _download_arm64_geckodriver(version="0.36.0"):
    """
    Download ARM64 geckodriver from GitHub releases.
    Returns the path to the downloaded geckodriver executable.
    """
    import urllib.request
    import zipfile
    import stat
    
    user_home = os.path.expanduser("~")
    driver_dir = os.path.join(user_home, ".wdm", "drivers", "geckodriver", "linux64", f"v{version}")
    geckodriver_path = os.path.join(driver_dir, "geckodriver")
    
    # If already exists and is ARM64, return it
    if os.path.exists(geckodriver_path):
        try:
            import subprocess
            result = subprocess.run(['file', geckodriver_path], capture_output=True, text=True)
            if 'aarch64' in result.stdout or 'arm64' in result.stdout:
                print(f"Using existing ARM64 geckodriver: {geckodriver_path}")
                return geckodriver_path
        except:
            pass
    
    # Download ARM64 geckodriver
    print(f"Downloading ARM64 geckodriver v{version}...")
    os.makedirs(driver_dir, exist_ok=True)
    
    # Geckodriver GitHub releases URL
    url = f"https://github.com/mozilla/geckodriver/releases/download/v{version}/geckodriver-v{version}-linux-aarch64.tar.gz"
    
    try:
        tar_path = os.path.join(driver_dir, "geckodriver.tar.gz")
        print(f"Downloading from: {url}")
        urllib.request.urlretrieve(url, tar_path)
        
        # Extract tar.gz
        import tarfile
        with tarfile.open(tar_path, 'r:gz') as tar:
            # Use filter='data' to avoid deprecation warning in Python 3.12+
            if hasattr(tarfile, 'data_filter'):
                tar.extractall(driver_dir, filter='data')
            else:
                tar.extractall(driver_dir)
        
        # Make executable
        if os.path.exists(geckodriver_path):
            os.chmod(geckodriver_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
            print(f"Downloaded ARM64 geckodriver: {geckodriver_path}")
            # Clean up tar file
            try:
                os.remove(tar_path)
            except:
                pass
            return geckodriver_path
        else:
            raise Exception("Geckodriver not found after extraction")
    except Exception as e:
        print(f"Failed to download ARM64 geckodriver: {e}")
        # Clean up on failure
        try:
            if os.path.exists(tar_path):
                os.remove(tar_path)
        except:
            pass
        raise

# --- FUNCTION TO GENERATE A NEW TOKEN ---
def generate_new_access_token():
    """
    Performs a full Selenium-based auto-login to generate and save a new
    access_token.txt file.
    """
    print("Starting Selenium auto-login to generate new access token...")
    try:
        # Use absolute path to ensure file is found regardless of working directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        api_key_path = os.path.join(script_dir, "key_secrets", "api_key.txt")
        api_key, api_secret, username, password, totp_secret = _load_key_secret_file(api_key_path)
        kite = KiteConnect(api_key=api_key)

        # Detect architecture and use appropriate browser
        arch = platform.machine().lower()  # Normalize to lowercase for comparison
        is_arm64 = (arch == 'aarch64' or arch == 'arm64')
        is_x86_64 = (arch == 'x86_64' or arch == 'amd64')
        
        driver = None
        browser_used = None
        firefox_error = None
        chrome_error = None
        
        # Strategy: Try Firefox first on ARM64, Chrome on x86_64
        # Fallback to the other if primary fails
        
        if is_arm64:
            # ARM64: Prefer Firefox (better ARM64 support)
            print(f"ARM64 architecture detected ({platform.machine()}) - attempting Firefox first")
            try:
                # For ARM64, webdriver_manager downloads wrong architecture, so download ARM64 directly
                try:
                    geckodriver_path = _download_arm64_geckodriver(version="0.36.0")
                except Exception as download_error:
                    print(f"[WARN] Failed to download ARM64 geckodriver: {download_error}")
                    # Try to find any existing ARM64 geckodriver
                    user_home = os.path.expanduser("~")
                    geckodriver_base = os.path.join(user_home, ".wdm", "drivers", "geckodriver", "linux64")
                    geckodriver_path = None
                    
                    if os.path.exists(geckodriver_base):
                        for version_dir in os.listdir(geckodriver_base):
                            potential_path = os.path.join(geckodriver_base, version_dir, "geckodriver")
                            if os.path.exists(potential_path):
                                try:
                                    import subprocess
                                    result = subprocess.run(['file', potential_path], capture_output=True, text=True)
                                    if 'aarch64' in result.stdout or 'arm64' in result.stdout:
                                        geckodriver_path = potential_path
                                        print(f"Found existing ARM64 geckodriver: {geckodriver_path}")
                                        break
                                except:
                                    pass
                    
                    if not geckodriver_path:
                        raise Exception(f"Could not find or download ARM64 geckodriver. Download error: {download_error}")
                
                # Check if Firefox browser is installed
                import shutil
                firefox_binary = shutil.which('firefox')
                if not firefox_binary:
                    # Try common Firefox paths
                    for path in ['/usr/bin/firefox', '/usr/local/bin/firefox', '/opt/firefox/firefox']:
                        if os.path.exists(path):
                            firefox_binary = path
                            break
                
                service = FirefoxService(geckodriver_path)
                options = webdriver.FirefoxOptions()
                
                if firefox_binary:
                    options.binary_location = firefox_binary
                    print(f"Using Firefox binary: {firefox_binary}")
                else:
                    print("[WARN] Firefox browser not found in PATH. Selenium will try to find it automatically.")
                    print("If Firefox fails to start, install it with: sudo yum install firefox -y")
                
                headless_env = os.getenv('HEADLESS', 'true').strip().lower()
                headless = headless_env in ('1', 'true', 'yes')
                if headless:
                    options.add_argument('--headless')
                print("Starting Firefox WebDriver...")
                driver = webdriver.Firefox(service=service, options=options)
                browser_used = 'Firefox'
                print("[OK] Firefox WebDriver started successfully.")
            except Exception as e:
                firefox_error = str(e)
                print(f"[WARN] Firefox failed: {firefox_error}")
                print("Falling back to Chrome (may fail on ARM64)...")
                driver = None
        
        if driver is None and is_x86_64:
            # x86_64: Prefer Chrome
            print(f"x86_64 architecture detected ({platform.machine()}) - using Chrome")
            try:
                chromedriver_path = ChromeDriverManager().install()
                # Suppress Chrome stderr output (DevTools messages)
                import subprocess
                service = ChromeService(chromedriver_path, log_output=subprocess.DEVNULL)
                options = webdriver.ChromeOptions()
                headless_env = os.getenv('HEADLESS', 'true').strip().lower()
                headless = headless_env in ('1', 'true', 'yes')
                if headless:
                    options.add_argument('--headless=new')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--disable-gpu')
                options.add_argument('--disable-extensions')
                options.add_argument('--disable-background-timer-throttling')
                options.add_argument('--disable-backgrounding-occluded-windows')
                options.add_argument('--disable-renderer-backgrounding')
                options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36')
                options.add_experimental_option('excludeSwitches', ['enable-logging', 'enable-automation'])
                options.add_experimental_option('useAutomationExtension', False)
                options.add_argument('--log-level=3')
                options.add_argument('--silent')
                options.add_argument('--disable-logging')
                # Suppress DevTools messages by disabling remote debugging completely
                options.add_argument('--remote-debugging-port=0')
                options.add_argument('--disable-dev-shm-usage')
                # Additional flags to suppress all Chrome output
                options.add_argument('--disable-logging')
                options.add_argument('--log-level=OFF')
                # Additional stability options for Windows
                options.add_argument('--disable-software-rasterizer')
                options.add_argument('--disable-web-security')
                options.add_argument('--allow-running-insecure-content')
                options.add_argument('--disable-features=IsolateOrigins,site-per-process')
                # Use a temporary profile to avoid profile lock issues
                temp_profile = tempfile.mkdtemp()
                options.add_argument(f'--user-data-dir={temp_profile}')
                
                print("Starting Chrome WebDriver...")
                driver = webdriver.Chrome(service=service, options=options)
                browser_used = 'Chrome'
                print("[OK] Chrome WebDriver started successfully.")
            except Exception as e:
                chrome_error = str(e)
                print(f"[WARN] Chrome failed: {chrome_error}")
                driver = None
        
        # Fallback: If primary browser failed, try the other
        if driver is None:
            if is_arm64:
                # ARM64 fallback: Try Chrome with ARM64 driver
                print("Attempting Chrome as fallback on ARM64...")
                try:
                    import subprocess
                    import urllib.request
                    import zipfile
                    import shutil
                    
                    # For ARM64, webdriver_manager downloads x86_64, so we need to download ARM64 manually
                    user_home = os.path.expanduser("~")
                    chromedriver_dir = os.path.join(user_home, ".wdm", "drivers", "chromedriver", "linux64")
                    chromedriver_path = None
                    
                    # Check if we already have an ARM64 chromedriver
                    for version_dir in os.listdir(chromedriver_dir) if os.path.exists(chromedriver_dir) else []:
                        potential_path = os.path.join(chromedriver_dir, version_dir, "chromedriver")
                        if os.path.exists(potential_path):
                            # Check if it's ARM64
                            try:
                                result = subprocess.run(['file', potential_path], capture_output=True, text=True)
                                if 'aarch64' in result.stdout or 'arm64' in result.stdout:
                                    chromedriver_path = potential_path
                                    print(f"Found existing ARM64 chromedriver: {chromedriver_path}")
                                    break
                            except:
                                pass
                    
                    # If no ARM64 driver found, try to download one
                    if not chromedriver_path:
                        print("No ARM64 chromedriver found. Attempting to download...")
                        # Note: Chrome for ARM64 Linux is limited. Try using the system chromedriver if available
                        system_chromedriver = shutil.which('chromedriver')
                        if system_chromedriver:
                            print(f"Using system chromedriver: {system_chromedriver}")
                            chromedriver_path = system_chromedriver
                        else:
                            # Fallback: try webdriver_manager (will likely fail, but worth trying)
                            print("Warning: No system chromedriver found. Trying webdriver_manager (may fail on ARM64)...")
                            chromedriver_path = ChromeDriverManager().install()
                    
                    # Suppress Chrome stderr output (DevTools messages)
                    service = ChromeService(chromedriver_path, log_output=subprocess.DEVNULL)
                    options = webdriver.ChromeOptions()
                    headless_env = os.getenv('HEADLESS', 'true').strip().lower()
                    headless = headless_env in ('1', 'true', 'yes')
                    if headless:
                        options.add_argument('--headless=new')
                    options.add_argument('--no-sandbox')
                    options.add_argument('--disable-dev-shm-usage')
                    options.add_argument('--disable-gpu')
                    options.add_argument('--log-level=3')
                    options.add_argument('--silent')
                    options.add_argument('--disable-logging')
                    options.add_argument('--remote-debugging-port=0')  # Disable remote debugging
                    options.add_experimental_option('excludeSwitches', ['enable-logging', 'enable-automation'])
                    options.add_experimental_option('useAutomationExtension', False)
                    # Additional stability options
                    options.add_argument('--disable-software-rasterizer')
                    options.add_argument('--disable-web-security')
                    options.add_argument('--allow-running-insecure-content')
                    options.add_argument('--disable-features=IsolateOrigins,site-per-process')
                    # Use a temporary profile to avoid profile lock issues
                    temp_profile = tempfile.mkdtemp()
                    options.add_argument(f'--user-data-dir={temp_profile}')
                    driver = webdriver.Chrome(service=service, options=options)
                    browser_used = 'Chrome (fallback)'
                    print("[OK] Chrome WebDriver started successfully (fallback).")
                except Exception as e:
                    chrome_error = chrome_error or str(e)
                    print(f"[ERROR] Chrome fallback also failed: {e}")
                    raise Exception(f"Could not start any browser on ARM64. Firefox error: {firefox_error}, Chrome error: {chrome_error}")
            elif is_x86_64:
                # x86_64 fallback: Try Firefox
                print("Attempting Firefox as fallback on x86_64...")
                try:
                    geckodriver_path = GeckoDriverManager().install()
                    service = FirefoxService(geckodriver_path)
                    options = webdriver.FirefoxOptions()
                    headless_env = os.getenv('HEADLESS', 'true').strip().lower()
                    headless = headless_env in ('1', 'true', 'yes')
                    if headless:
                        options.add_argument('--headless')
                    driver = webdriver.Firefox(service=service, options=options)
                    browser_used = 'Firefox (fallback)'
                    print("[OK] Firefox WebDriver started successfully (fallback).")
                except Exception as e:
                    firefox_error = firefox_error or str(e)
                    print(f"[ERROR] Firefox fallback also failed: {e}")
                    raise Exception(f"Could not start any browser on x86_64. Chrome error: {chrome_error}, Firefox error: {firefox_error}")
            else:
                actual_arch = platform.machine()
                raise Exception(f"Unsupported architecture: {actual_arch}. Supported: aarch64/arm64, x86_64/amd64")
        
        if driver is None:
            raise Exception("Failed to initialize any browser driver")
        
        print(f"[OK] Using {browser_used} on {platform.machine()} architecture")

        try:
            login_url = kite.login_url()
            print(f"Navigating to login URL: {login_url}")
            driver.get(login_url)
            driver.implicitly_wait(10)
            print("[OK] Page loaded successfully")
        except Exception as e:
            raise Exception(f"Failed to navigate to login page: {e}")

        try:
            print("Waiting for username input field...")
            username_input = WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.XPATH, '//input[@type="text"]')))
            print("Waiting for password input field...")
            password_input = WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.XPATH, '//input[@type="password"]')))
            print("[OK] Login form fields found")
            username_input.send_keys(username)
            password_input.send_keys(password)
            print("[OK] Credentials entered")
        except Exception as e:
            raise Exception(f"Failed to find or fill login form: {e}")
        
        try:
            print("Clicking submit button...")
            WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//button[@type="submit"]'))).click()
            print("[OK] Submit button clicked")
        except Exception as e:
            raise Exception(f"Failed to click submit button: {e}")
        
        try:
            print("Waiting for TOTP pin input field...")
            pin = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '//input[@type="number"]')))
            totp = TOTP(totp_secret)
            token = totp.now()
            print(f"Entering TOTP token: {token[:2]}**")
            pin.send_keys(token)
            print("[OK] TOTP token entered")
        except Exception as e:
            raise Exception(f"Failed to enter TOTP token: {e}")

        try:
            print("Waiting for redirect URL with request_token...")
            WebDriverWait(driver, 20).until(EC.url_contains("request_token="))
            redirect_url = driver.current_url or ""
            print(f"Redirect URL received: {redirect_url[:100]}...")
            if "request_token=" not in redirect_url:
                raise RuntimeError("Login redirect did not contain request_token; check credentials or UI changes.")
            request_token = redirect_url.split('request_token=')[1].split('&')[0]
            print(f"[OK] Request token extracted: {request_token[:10]}...")
        except Exception as e:
            raise Exception(f"Failed to get request token from redirect: {e}")

        try:
            print("Generating session with request token...")
            data = kite.generate_session(request_token, api_secret=api_secret)
            access_token = data["access_token"]
            access_token_path = os.path.join(script_dir, "key_secrets", "access_token.txt")
            with open(access_token_path, 'w') as file:
                file.write(access_token)
            print("SUCCESS: New access_token.txt has been generated and saved.")
            
            # Save to cache immediately after generation
            try:
                profile = kite.profile()
                user_id = profile['user_id']
                _save_token_cache(script_dir, access_token, api_key, user_id)
                print(f"Token cached for user: {user_id}")
            except Exception as cache_e:
                print(f"Warning: Could not cache token: {cache_e}")
            
            return True
        except Exception as e:
            raise Exception(f"Failed to generate session or save access token: {e}")

    except WebDriverException as e:
        # Handle Selenium-specific exceptions
        import traceback
        error_msg = str(e) if str(e) else repr(e)
        error_type = type(e).__name__
        print(f"FATAL: Selenium WebDriver error: {error_type}: {error_msg}")
        if hasattr(e, 'msg') and e.msg:
            print(f"WebDriver message: {e.msg}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False
    except TimeoutException as e:
        # Handle timeout exceptions specifically
        import traceback
        error_msg = str(e) if str(e) else "Timeout waiting for element"
        print(f"FATAL: Timeout error: {error_msg}")
        print("This usually means the page took too long to load or an element was not found.")
        print("\nFull traceback:")
        traceback.print_exc()
        return False
    except NoSuchElementException as e:
        # Handle element not found exceptions
        import traceback
        error_msg = str(e) if str(e) else "Element not found"
        print(f"FATAL: Element not found: {error_msg}")
        print("This usually means the page structure has changed or the element selector is incorrect.")
        print("\nFull traceback:")
        traceback.print_exc()
        return False
    except Exception as e:
        # Handle all other exceptions
        import traceback
        try:
            error_msg = str(e) if str(e) else repr(e)
            error_type = type(e).__name__
        except UnicodeEncodeError:
            error_msg = repr(e)  # Use repr() as fallback for encoding issues
            error_type = type(e).__name__
        
        # Print detailed error information
        print(f"FATAL: Selenium auto-login failed: {error_type}: {error_msg}")
        print("\nFull traceback:")
        try:
            traceback.print_exc()
        except Exception:
            # If traceback printing fails, at least show the exception type and message
            print(f"Exception type: {error_type}")
            print(f"Exception message: {error_msg}")
            if hasattr(e, '__traceback__') and e.__traceback__:
                import sys
                traceback.print_exception(type(e), e, e.__traceback__, file=sys.stdout)
        return False
    finally:
        if 'driver' in locals() and driver:
            driver.quit()


# --- MAIN FUNCTION FOR OTHER SCRIPTS TO USE ---
def get_kite_client(silent: bool = False):
    """
    Tries to authenticate with an existing token. Uses cache to reduce API calls.
    If cache is valid, returns cached client without API validation.
    If cache is invalid or missing, validates and updates cache.
    If validation fails, automatically runs the Selenium process to generate a new one.

    Args:
        silent (bool): When True, suppresses informational print statements
                       while still surfacing warnings/errors.
    """
    # Use absolute path to ensure file is found regardless of working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    api_key_path = os.path.join(script_dir, "key_secrets", "api_key.txt")
    
    try:
        api_key, api_secret, username, password, totp_secret = _load_key_secret_file(api_key_path)
    except Exception as e:
        print(f"FATAL: Could not load API credentials: {e}")
        sys.exit(1)
    
    # Check cache first
    cache_data = _load_token_cache(script_dir)
    if cache_data and _is_token_cache_valid(cache_data):
        # Use cached token without API validation
        cached_token = cache_data.get('access_token')
        cached_api_key = cache_data.get('api_key')
        
        # Verify API key matches (in case credentials changed)
        if cached_api_key == api_key and cached_token:
            try:
                kite = KiteConnect(api_key=api_key)
                kite.set_access_token(cached_token)
                
                if not silent:
                    user_id = cache_data.get('user_id', 'cached')
                    print(f"SUCCESS: Using cached token for user: {user_id} (no API call)")
                
                # Update cache with current time (but don't validate)
                _save_token_cache(script_dir, cached_token, api_key, user_id)
                return kite
            except Exception as e:
                # Cache token failed, fall through to validation
                if not silent:
                    print(f"Cache token failed, validating: {e}")
    
    # Cache invalid or missing - validate token
    if not silent:
        print("Attempting to authenticate Kite client...")
    
    try:
        access_token_path = os.path.join(script_dir, "key_secrets", "access_token.txt")
        with open(access_token_path, 'r') as f:
            access_token = f.readline().strip()

        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        
        # The crucial test: verify the token is valid.
        profile = kite.profile()
        user_id = profile['user_id']
        
        # Save to cache after successful validation
        _save_token_cache(script_dir, access_token, api_key, user_id)
        
        if not silent:
            print(f"SUCCESS: Authenticated using existing token for user: {user_id}")
        return kite

    except (FileNotFoundError, Exception) as e:
        print(f"Could not authenticate with existing token (Reason: {e}).")
        print("Attempting to generate a new token automatically...")
        
        # --- Recovery Step ---
        if generate_new_access_token():
            if not silent:
                print("Retrying authentication with the new token...")
            # After successful generation, we try one more time.
            try:
                access_token_path = os.path.join(script_dir, "key_secrets", "access_token.txt")
                with open(access_token_path, 'r') as f:
                    access_token = f.readline().strip()
                kite = KiteConnect(api_key=api_key)
                kite.set_access_token(access_token)
                profile = kite.profile()
                user_id = profile['user_id']
                
                # Save new token to cache
                _save_token_cache(script_dir, access_token, api_key, user_id)
                
                if not silent:
                    print(f"SUCCESS: Authenticated with newly generated token for user: {user_id}")
                return kite
            except Exception as final_e:
                print(f"FATAL: Failed to authenticate even with a new token. {final_e}")
                sys.exit(1)
        else:
            print("FATAL: Could not generate a new access token. Exiting.")
            sys.exit(1)


# --- ENTRY POINT FOR MANUAL EXECUTION ---
if __name__ == '__main__':
    # You can still run this manually to force a new token generation if needed.
    print("Manual token generation initiated.")
    generate_new_access_token()