"""
Async API Server for Trading System
Replaces Flask with aiohttp for non-blocking command handling
"""

import asyncio
import logging
from aiohttp import web, ClientTimeout
import json
from typing import Dict, Any, Optional
from threading import Lock

from event_system import Event, EventType, get_event_dispatcher

# Logger setup
logger = logging.getLogger(__name__)

class AsyncAPIServer:
    """
    Async API server using aiohttp to handle commands and configuration updates
    without blocking the event loop.
    """
    def __init__(self, host: str = 'localhost', port: int = 5000):
        self.host = host
        self.port = port
        self.event_dispatcher = get_event_dispatcher()
        self.app = web.Application()
        self.runner = None
        self.site = None

        # Dynamic config for TRADE_SETTINGS (thread-safe)
        self.dynamic_trade_settings: Dict[str, Any] = {}
        self.config_lock = asyncio.Lock()

        # Setup routes
        self._setup_routes()

        logger.info(f"Async API server initialized for {host}:{port}")

    def _setup_routes(self):
        """Setup API routes"""
        self.app.router.add_post('/command', self.handle_command)
        self.app.router.add_post('/set_sentiment_mode', self.handle_set_sentiment_mode)
        self.app.router.add_post('/update_config', self.handle_update_config)
        self.app.router.add_get('/status', self.handle_status)
        self.app.router.add_get('/health', self.handle_health)
        self.app.router.add_get('/get_config', self.handle_get_config)

    async def handle_command(self, request: web.Request) -> web.Response:
        """Handle trading commands"""
        try:
            data = await request.json()
            command = data.get('sentiment')

            if not command:
                return web.json_response({
                    "status": "error",
                    "message": "No command provided"
                }, status=400)

            logger.info(f"Received async command: {command}")

            # Validate command
            valid_commands = ["BULLISH", "BEARISH", "NEUTRAL", "DISABLE", "BUY_CE", "BUY_PE", "FORCE_EXIT", "FORCE_EXIT_CE", "FORCE_EXIT_PE"]
            if command not in valid_commands:
                return web.json_response({
                    "status": "error",
                    "message": f"Invalid command. Valid commands: {valid_commands}"
                }, status=400)

            # Dispatch command as event
            self.event_dispatcher.dispatch_event(
                Event(EventType.USER_COMMAND, {
                    'command': command,
                    'timestamp': asyncio.get_event_loop().time(),
                    'source': 'api_server'
                }, source='api_server')
            )

            # For immediate actions, return success immediately
            if command in ["BUY_CE", "BUY_PE", "FORCE_EXIT", "FORCE_EXIT_CE", "FORCE_EXIT_PE"]:
                return web.json_response({
                    "status": "success",
                    "message": f"Command '{command}' queued for immediate execution"
                })

            # For sentiment changes, return success
            # Note: DISABLE command is now a toggle
            if command == 'DISABLE':
                return web.json_response({
                    "status": "success",
                    "message": f"Autonomous trades toggle command sent (DISABLE)"
                })
            return web.json_response({
                "status": "success",
                "message": f"Sentiment set to {command}"
            })

        except json.JSONDecodeError:
            return web.json_response({
                "status": "error",
                "message": "Invalid JSON"
            }, status=400)
        except Exception as e:
            logger.error(f"Error handling command: {e}")
            return web.json_response({
                "status": "error",
                "message": f"Internal error: {str(e)}"
            }, status=500)

    async def handle_set_sentiment_mode(self, request: web.Request) -> web.Response:
        """Handle sentiment mode setting"""
        try:
            data = await request.json()
            mode = data.get('mode')
            manual_sentiment = data.get('manual_sentiment')

            if not mode:
                return web.json_response({
                    "status": "error",
                    "message": "No mode provided"
                }, status=400)

            mode = mode.upper()
            valid_modes = ["AUTO", "MANUAL", "DISABLE"]
            if mode not in valid_modes:
                return web.json_response({
                    "status": "error",
                    "message": f"Invalid mode. Valid modes: {valid_modes}"
                }, status=400)

            # Validate manual_sentiment if mode is MANUAL
            if mode == "MANUAL":
                if not manual_sentiment:
                    return web.json_response({
                        "status": "error",
                        "message": "manual_sentiment is required when mode=MANUAL"
                    }, status=400)
                manual_sentiment = manual_sentiment.upper()
                valid_sentiments = ["BULLISH", "BEARISH", "NEUTRAL", "DISABLE"]
                if manual_sentiment not in valid_sentiments:
                    return web.json_response({
                        "status": "error",
                        "message": f"Invalid manual_sentiment. Valid sentiments: {valid_sentiments}"
                    }, status=400)

            logger.info(f"Received set_sentiment_mode request: mode={mode}, manual_sentiment={manual_sentiment}")

            # Dispatch command as event
            self.event_dispatcher.dispatch_event(
                Event(EventType.USER_COMMAND, {
                    'command': 'SET_SENTIMENT_MODE',
                    'mode': mode,
                    'manual_sentiment': manual_sentiment,
                    'timestamp': asyncio.get_event_loop().time(),
                    'source': 'api_server'
                }, source='api_server')
            )

            return web.json_response({
                "status": "success",
                "message": f"Sentiment mode set to {mode}" + (f" with sentiment {manual_sentiment}" if manual_sentiment else ""),
                "mode": mode,
                "sentiment": manual_sentiment if mode == "MANUAL" else None
            })

        except json.JSONDecodeError:
            return web.json_response({
                "status": "error",
                "message": "Invalid JSON"
            }, status=400)
        except Exception as e:
            logger.error(f"Error handling set_sentiment_mode: {e}")
            return web.json_response({
                "status": "error",
                "message": f"Internal error: {str(e)}"
            }, status=500)

    async def handle_update_config(self, request: web.Request) -> web.Response:
        """Handle configuration updates"""
        try:
            data = await request.json()

            if not data:
                return web.json_response({
                    "status": "error",
                    "message": "No data provided"
                }, status=400)

            async with self.config_lock:
                updated_keys = []
                for key, value in data.items():
                    if key in self.dynamic_trade_settings:
                        old_value = self.dynamic_trade_settings[key]
                        self.dynamic_trade_settings[key] = value
                        updated_keys.append(f"{key}: {old_value} -> {value}")
                        logger.info(f"Updated config {key} from {old_value} to {value}")
                    else:
                        return web.json_response({
                            "status": "error",
                            "message": f"Invalid config key: {key}"
                        }, status=400)

                # Dispatch config update event
                self.event_dispatcher.dispatch_event(
                    Event(EventType.CONFIG_UPDATE, {
                        'updates': data,
                        'updated_keys': updated_keys,
                        'timestamp': asyncio.get_event_loop().time()
                    }, source='api_server')
                )

            return web.json_response({
                "status": "success",
                "message": "Config updated successfully",
                "updated_keys": updated_keys
            })

        except json.JSONDecodeError:
            return web.json_response({
                "status": "error",
                "message": "Invalid JSON"
            }, status=400)
        except Exception as e:
            logger.error(f"Error handling config update: {e}")
            return web.json_response({
                "status": "error",
                "message": f"Internal error: {str(e)}"
            }, status=500)
    
    async def handle_get_config(self, request: web.Request) -> web.Response:
        """Handle get config requests for specific keys"""
        try:
            key = request.query.get('key')
            if not key:
                return web.json_response({
                    "status": "error",
                    "message": "Missing 'key' parameter"
                }, status=400)
            
            async with self.config_lock:
                value = self.dynamic_trade_settings.get(key)
            
            return web.json_response({
                "status": "success",
                "key": key,
                "value": value
            })
        except Exception as e:
            logger.error(f"Error handling get config: {e}")
            return web.json_response({
                "status": "error",
                "message": f"Internal error: {str(e)}"
            }, status=500)

    async def handle_status(self, request: web.Request) -> web.Response:
        """Handle status requests"""
        try:
            # Get event system status
            event_status = {
                'queue_size': self.event_dispatcher.get_queue_size(),
                'registered_handlers': self.event_dispatcher.get_registered_handlers()
            }

            # Get config status
            async with self.config_lock:
                config_status = self.dynamic_trade_settings.copy()

            return web.json_response({
                "status": "success",
                "event_system": event_status,
                "config": config_status,
                "server": {
                    "host": self.host,
                    "port": self.port,
                    "running": self.site is not None
                }
            })

        except Exception as e:
            logger.error(f"Error handling status request: {e}")
            return web.json_response({
                "status": "error",
                "message": f"Internal error: {str(e)}"
            }, status=500)

    async def handle_health(self, request: web.Request) -> web.Response:
        """Handle health check requests"""
        return web.json_response({
            "status": "healthy",
            "service": "async_api_server",
            "timestamp": asyncio.get_event_loop().time()
        })

    async def initialize_config(self, initial_config: Dict[str, Any]):
        """Initialize dynamic configuration"""
        async with self.config_lock:
            self.dynamic_trade_settings.update(initial_config)
            logger.info(f"Initialized config with {len(initial_config)} settings")

    async def get_config_value(self, key: str) -> Any:
        """Get a configuration value"""
        async with self.config_lock:
            return self.dynamic_trade_settings.get(key)

    async def update_config_value(self, key: str, value: Any):
        """Update a single configuration value"""
        async with self.config_lock:
            old_value = self.dynamic_trade_settings.get(key)
            self.dynamic_trade_settings[key] = value

            # Dispatch config update event
            self.event_dispatcher.dispatch_event(
                Event(EventType.CONFIG_UPDATE, {
                    'updates': {key: value},
                    'updated_keys': [f"{key}: {old_value} -> {value}"],
                    'timestamp': asyncio.get_event_loop().time()
                }, source='api_server')
            )

            logger.info(f"Updated config {key} from {old_value} to {value}")

    async def start(self):
        """Start the async API server"""
        try:
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()

            self.site = web.TCPSite(self.runner, self.host, self.port)
            await self.site.start()

            logger.info(f"[OK] Async API server started on {self.host}:{self.port}")

            # Dispatch system startup event
            self.event_dispatcher.dispatch_event(
                Event(EventType.SYSTEM_STARTUP, {
                    'message': f'API server started on {self.host}:{self.port}',
                    'host': self.host,
                    'port': self.port
                }, source='api_server')
            )

        except Exception as e:
            logger.error(f"Failed to start async API server: {e}")
            self.event_dispatcher.dispatch_event(
                Event(EventType.ERROR_OCCURRED, {
                    'message': f"api_server_startup_error: {str(e)}"
                }, source='api_server')
            )
            raise

    async def stop(self):
        """Stop the async API server"""
        try:
            if self.site:
                await self.site.stop()
                logger.info("Async API server stopped")

            if self.runner:
                await self.runner.cleanup()

            # Dispatch system shutdown event
            self.event_dispatcher.dispatch_event(
                Event(EventType.SYSTEM_SHUTDOWN, {
                    'message': 'API server shutdown'
                }, source='api_server')
            )

        except Exception as e:
            logger.error(f"Error stopping async API server: {e}")

    def is_running(self) -> bool:
        """Check if the server is running"""
        return self.site is not None and self.runner is not None


# Global instance
async_api_server = AsyncAPIServer()


def get_async_api_server() -> AsyncAPIServer:
    """Get the global async API server instance"""
    return async_api_server


# Example usage and testing functions
async def test_api_server():
    """Test function for the async API server"""
    logger.info("Testing async API server...")

    # Test health endpoint
    async with ClientTimeout(total=10):
        async with web.ClientSession() as session:
            try:
                async with session.get('http://localhost:5000/health') as resp:
                    result = await resp.json()
                    logger.info(f"Health check result: {result}")
            except Exception as e:
                logger.error(f"Health check failed: {e}")

    # Test command endpoint
    test_commands = [
        {"sentiment": "BULLISH"},
        {"sentiment": "BUY_CE"},
        {"sentiment": "DISABLE"}
    ]

    for command in test_commands:
        try:
            async with session.post('http://localhost:5000/command',
                                  json=command) as resp:
                result = await resp.json()
                logger.info(f"Command {command} result: {result}")
        except Exception as e:
            logger.error(f"Command test failed: {e}")

    logger.info("API server testing completed")
