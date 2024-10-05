# main.py

import logging
import traceback

from logger import logger
from interface.gradio_interface import create_gradio_interface

def main():
    """
    Main function to launch the Gradio interface.
    """
    try:
        logger.info("Launching Gradio interface.")
        iface = create_gradio_interface()
        iface.launch(share=False)  # Set share=True to create a public link
    except Exception as e:
        logger.error(f"Failed to launch Gradio interface: {str(e)}")
        logger.debug(traceback.format_exc())

if __name__ == "__main__":
    main()
