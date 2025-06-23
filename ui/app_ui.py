import gradio as gr

def respond(message):
    return chat_with_bot(message)

interface = gr.ChatInterface(fn=respond, title="🎵 MusicBot")
interface.launch()