import gradio as gr
from gradio.components import Component

from src.webui.webui_manager import WebuiManager
from src.utils import config


def create_browser_use_agent_tab(webui_manager: WebuiManager) -> dict[str, Component]:
    """
    Create the run agent tab
    """
    input_components = set(webui_manager.get_components())
    tab_components = {}

    chatbot = gr.Chatbot(type='messages', label="Chat History", height=600)
    user_input = gr.Textbox(
        label="User Input",
        lines=3,
        value="go to google.com and type 'OpenAI' click search and give me the first url",
        interactive=True
    )

    with gr.Row():
        stop_button = gr.Button("⏹️ Stop", interactive=False, variant="stop", scale=2)
        clear_button = gr.Button("🧹 Clear", interactive=False, variant="stop", scale=2)
        run_button = gr.Button("▶️ Summit", variant="primary", scale=3)

    browser_view = gr.HTML(
        value="<h1 style='width:80vw; height:50vh'>Waiting for browser session...</h1>",
        label="Browser Live View",
        visible=False
    )

    with gr.Row():
        agent_final_result = gr.Textbox(
            label="Final Result", lines=3, show_label=True, interactive=False
        )
        agent_errors = gr.Textbox(
            label="Errors", lines=3, show_label=True, interactive=False
        )

    with gr.Row():
        agent_trace_file = gr.File(label="Trace File", interactive=False)
        agent_history_file = gr.File(label="Agent History", interactive=False)

    recording_gif = gr.Image(label="Result GIF", format="gif", interactive=False)
    tab_components.update(
        dict(
            chatbot=chatbot,
            user_input=user_input,
            clear_button=clear_button,
            run_button=run_button,
            stop_button=stop_button,
            agent_final_result=agent_final_result,
            agent_errors=agent_errors,
            agent_trace_file=agent_trace_file,
            agent_history_file=agent_history_file,
            recording_gif=recording_gif,
            browser_view=browser_view
        )
    )
    return tab_components
