import gradio as gr
from gradio.components import Component

from src.webui.webui_manager import WebuiManager
from src.utils import config


def create_deep_research_agent_tab(webui_manager: WebuiManager) -> dict[str, Component]:
    """
    Creates a deep research agent tab
    """
    input_components = set(webui_manager.get_components())
    tab_components = {}

    research_task = gr.Textbox(label="Research Task", lines=5,
                               value="Give me a detailed plan for traveling to Switzerland on June 1st.",
                               interactive=True)
    with gr.Row():
        max_iteration = gr.Number(label="Max Search Iteration", value=3,
                                  precision=0,
                                  interactive=True)  # precision=0 确保是整数
        max_query = gr.Number(label="Max Query per Iteration", value=1,
                              precision=0,
                              interactive=True)  # precision=0 确保是整数
    with gr.Row():
        stop_button = gr.Button("⏹️ Stop", variant="stop", scale=2)
        start_button = gr.Button("▶️ Run", variant="primary", scale=3)
    markdown_display = gr.Markdown(label="Research Report")
    markdown_download = gr.File(label="Download Research Report", interactive=False)
    tab_components.update(
        dict(
            research_task=research_task,
            max_iteration=max_iteration,
            max_query=max_query,
            start_button=start_button,
            stop_button=stop_button,
            markdown_display=markdown_display,
            markdown_download=markdown_download,
        )
    )
    return tab_components
