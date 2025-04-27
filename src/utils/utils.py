import base64
import os
import time
from pathlib import Path
from typing import Dict, Optional
import requests
import json
import gradio as gr
import uuid


# Callback to update the model name dropdown based on the selected provider
def update_model_dropdown(llm_provider, api_key=None, base_url=None):
    """
    Update the model name dropdown with predefined models for the selected provider.
    """
    import gradio as gr
    # Use API keys from .env if not provided
    if not api_key:
        api_key = os.getenv(f"{llm_provider.upper()}_API_KEY", "")
    if not base_url:
        base_url = os.getenv(f"{llm_provider.upper()}_BASE_URL", "")

    # Use predefined models for the selected provider
    if llm_provider in model_names:
        return gr.Dropdown(choices=model_names[llm_provider], value=model_names[llm_provider][0], interactive=True)
    else:
        return gr.Dropdown(choices=[], value="", interactive=True, allow_custom_value=True)


def encode_image(img_path):
    if not img_path:
        return None
    with open(img_path, "rb") as fin:
        image_data = base64.b64encode(fin.read()).decode("utf-8")
    return image_data


def get_latest_files(directory: str, file_types: list = ['.webm', '.zip']) -> Dict[str, Optional[str]]:
    """Get the latest recording and trace files"""
    latest_files: Dict[str, Optional[str]] = {ext: None for ext in file_types}

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return latest_files

    for file_type in file_types:
        try:
            matches = list(Path(directory).rglob(f"*{file_type}"))
            if matches:
                latest = max(matches, key=lambda p: p.stat().st_mtime)
                # Only return files that are complete (not being written)
                if time.time() - latest.stat().st_mtime > 1.0:
                    latest_files[file_type] = str(latest)
        except Exception as e:
            print(f"Error getting latest {file_type} file: {e}")

    return latest_files


async def capture_screenshot(browser_context):
    """Capture and encode a screenshot"""
    # Extract the Playwright browser instance
    playwright_browser = browser_context.browser.playwright_browser  # Ensure this is correct.

    # Check if the browser instance is valid and if an existing context can be reused
    if playwright_browser and playwright_browser.contexts:
        playwright_context = playwright_browser.contexts[0]
    else:
        return None

    # Access pages in the context
    pages = None
    if playwright_context:
        pages = playwright_context.pages

    # Use an existing page or create a new one if none exist
    if pages:
        active_page = pages[0]
        for page in pages:
            if page.url != "about:blank":
                active_page = page
    else:
        return None

    # Take screenshot
    try:
        screenshot = await active_page.screenshot(
            type='jpeg',
            quality=75,
            scale="css"
        )
        encoded = base64.b64encode(screenshot).decode('utf-8')
        return encoded
    except Exception as e:
        return None


class ConfigManager:
    def __init__(self):
        self.components = {}
        self.component_order = []

    def register_component(self, name: str, component):
        """Register a gradio component for config management."""
        self.components[name] = component
        if name not in self.component_order:
            self.component_order.append(name)
        return component

    def save_current_config(self):
        """Save the current configuration of all registered components."""
        current_config = {}
        for name in self.component_order:
            component = self.components[name]
            # Get the current value from the component
            current_config[name] = getattr(component, "value", None)

        return save_config_to_file(current_config)

    def update_ui_from_config(self, config_file):
        """Update UI components from a loaded configuration file."""
        if config_file is None:
            return [gr.update() for _ in self.component_order] + ["No file selected."]

        loaded_config = load_config_from_file(config_file.name)

        if not isinstance(loaded_config, dict):
            return [gr.update() for _ in self.component_order] + ["Error: Invalid configuration file."]

        # Prepare updates for all components
        updates = []
        for name in self.component_order:
            if name in loaded_config:
                updates.append(gr.update(value=loaded_config[name]))
            else:
                updates.append(gr.update())

        updates.append("Configuration loaded successfully.")
        return updates

    def get_all_components(self):
        """Return all registered components in the order they were registered."""
        return [self.components[name] for name in self.component_order]


def load_config_from_file(config_file):
    """Load settings from a config file (JSON format)."""
    try:
        with open(config_file, 'r') as f:
            settings = json.load(f)
        return settings
    except Exception as e:
        return f"Error loading configuration: {str(e)}"


def save_config_to_file(settings, save_dir="./tmp/webui_settings"):
    """Save the current settings to a UUID.json file with a UUID name."""
    os.makedirs(save_dir, exist_ok=True)
    config_file = os.path.join(save_dir, f"{uuid.uuid4()}.json")
    with open(config_file, 'w') as f:
        json.dump(settings, f, indent=2)
    return f"Configuration saved to {config_file}"
