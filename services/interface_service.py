import gradio as gr

def create_gradio_interface(process_input_fn):
    """
    Defines and returns the Gradio Blocks interface with enhanced layout, focusing on YouTube link input.
    """
    with gr.Blocks() as iface:
        # Improved header markdown with larger font and an appealing subheading
        gr.Markdown("""
        <h1 style='font-size: 2.5em; text-align: center;'>🚁 PIX4D-ARCHSCAN</h1>
        <p style='font-size: 1.2em; text-align: center;'>Upload a YouTube video link, specify the number of frames to extract, and Pixtral will provide live summarizations of the content in a larger chat interface. The frames and their summaries will be saved automatically.</p>
        """, elem_id="header")

        # Layout for YouTube video link input
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=300):
                youtube_link = gr.Textbox(label="📥 YouTube Video Link", placeholder="Enter YouTube video URL here", visible=True)
                frame_number_input = gr.Number(label="🔢 Frame Number to Extract", value=50, precision=0, step=1, interactive=True)
                include_images_input = gr.Checkbox(label="🖼️ Include Frame Images in Summaries", value=False)
                submit_btn = gr.Button("▶️ Process Video", elem_classes="submit-btn")
                
            # Expanded scale to make the chatbot area larger
            with gr.Column(scale=5, min_width=600):
                chatbot = gr.Chatbot(label="💬 Live Summarization", elem_id="chatbot", height=500)  # Larger height for better content display

        # Hidden states to store chat history and frame summaries
        chat_history = gr.State(value=[])
        frame_summaries = gr.State(value=[])

        # Define the generator function for streaming summaries to the chatbot
        submit_btn.click(
            fn=process_input_fn,  # Accept the processing function as a parameter
            inputs=[youtube_link, frame_number_input, include_images_input, chat_history],
            outputs=[chatbot, frame_summaries],
            show_progress=True
        )

    return iface
