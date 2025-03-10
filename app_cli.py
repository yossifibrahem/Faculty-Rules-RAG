"""
LLM Tool Calling Web Application
This module provides a chat interface with various tool-calling capabilities.
"""

# Standard library imports
import json
import os
import shutil
from datetime import datetime
from typing import List, Dict, Tuple, Any
from textwrap import fill

# Third-party imports
from openai import OpenAI
from colorama import init, Fore, Back, Style

# Local imports
from SearchRules import search_rules as search_info

# Constants
MODEL = "qwen2.5-3b-instruct"
BASE_URL = "http://127.0.0.1:1234/v1"
API_KEY = "lm-studio"

# Initialize OpenAI client
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Initialize colorama
init()

# Tool definitions
Tools = [{
    "type": "function",
    "function": {
        "name": "search_info",
        "description": (
            "Use this every time the stuedent asks a question that require looking for."
            "This tool searches the faculty rules and regulations for the given query."
            "The tool returns the relevant information found in the rules."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "describe what the user is looking for in details."}
            },
            "required": ["query"]
        }
    }
}]

def get_terminal_width() -> int:
    """Get the current terminal width."""
    width, _ = shutil.get_terminal_size()
    return width

def create_centered_box(text: str, padding: int = 4) -> str:
    """Create a centered box with dynamic width."""
    width = get_terminal_width()
    content_width = width - 2  # Account for borders
    lines = text.split('\n')
    
    box = '╔' + '═' * (width - 2) + '╗\n'
    box += '║' + ' ' * (width - 2) + '║\n'
    
    for line in lines:
        if line.strip():
            padded_line = line.center(width - 2)
            box += '║' + padded_line + '║\n'
    
    box += '║' + ' ' * (width - 2) + '║\n'
    box += '╚' + '═' * (width - 2) + '╝'
    return box

def process_stream(stream: Any, add_assistant_label: bool = True) -> Tuple[str, List[Dict]]:
    """
    Handle streaming responses from the API.
    """
    collected_text = ""
    tool_calls = []
    first_chunk = True

    for chunk in stream:
        delta = chunk.choices[0].delta

        # Handle regular text output
        if delta.content:
            if first_chunk:
                print()
                if add_assistant_label:
                    print(f"{Fore.BLUE}Assistant:{Style.RESET_ALL}", end=" ", flush=True)
                first_chunk = False
            print(delta.content, end="", flush=True)
            collected_text += delta.content

        # Handle tool calls
        elif delta.tool_calls:
            if len(tool_calls) <= delta.tool_calls[0].index:
                tool_calls.append({
                    "id": "", "type": "function",
                    "function": {"name": "", "arguments": ""}
                })
            tool_calls[delta.tool_calls[0].index] = {
                "id": (tool_calls[delta.tool_calls[0].index]["id"] + (delta.tool_calls[0].id or "")),
                "type": "function",
                "function": {
                    "name": (tool_calls[delta.tool_calls[0].index]["function"]["name"] + (delta.tool_calls[0].function.name or "")),
                    "arguments": (tool_calls[delta.tool_calls[0].index]["function"]["arguments"] + (delta.tool_calls[0].function.arguments or ""))
                }
            }
        
    return collected_text, tool_calls

def process_non_stream(response: Any, add_assistant_label: bool = True) -> Tuple[str, List[Dict]]:
    """
    Handle non-streaming responses from the API.
    
    Args:
        response: The non-streaming response from the API
        add_assistant_label: Whether to prefix output with 'Assistant:'
    
    Returns:
        Tuple containing response text and tool calls
    """
    collected_text = ""
    tool_calls = []
    
    print()
    if add_assistant_label:
        print(f"{Fore.BLUE}Assistant:{Style.RESET_ALL}", end=" ", flush=True)
    
    # Extract content if present
    if response.choices[0].message.content:
        content = response.choices[0].message.content
        print(content, end="", flush=True)
        collected_text = content
    
    # Extract tool calls if present
    if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
        for tc in response.choices[0].message.tool_calls:
            tool_calls.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            })
    
    return collected_text, tool_calls

def show_help() -> None:
    """Display available tools and commands."""
    width = get_terminal_width()
    
    print(f"\n{Back.BLUE} Available Tools {Style.RESET_ALL}")
    print("─" * width)
    for tool in Tools:
        name = f"{Fore.BLUE}• {tool['function']['name']}{Style.RESET_ALL}"
        desc = tool['function']['description']
        wrapped_desc = fill(desc, width=width - len(name) + len(Fore.BLUE) + len(Style.RESET_ALL))
        print(f"{name}: {wrapped_desc}")
    
    print(f"\n{Back.BLUE} Available Commands {Style.RESET_ALL}")
    print("─" * width)
    print(f"{Fore.BLUE}• clear{Style.RESET_ALL}: Clear the chat history")
    print(f"{Fore.BLUE}• help{Style.RESET_ALL}: Show this help message")

def display_welcome_banner() -> None:
    """Display a styled 3D welcome banner."""
    banner = """
     █████╗ ██╗    █████╗ ███████╗███████╗██╗███████╗████████╗ █████╗ ███╗   ██╗████████╗
    ██╔══██╗██║   ██╔══██╗██╔════╝██╔════╝██║██╔════╝╚══██╔══╝██╔══██╗████╗  ██║╚══██╔══╝
    ███████║██║   ███████║███████╗███████╗██║███████╗   ██║   ███████║██╔██╗ ██║   ██║   
    ██╔══██║██║   ██╔══██║╚════██║╚════██║██║╚════██║   ██║   ██╔══██║██║╚██╗██║   ██║   
    ██║  ██║██║   ██║  ██║███████║███████║██║███████║   ██║   ██║  ██║██║ ╚████║   ██║   
    ╚═╝  ╚═╝╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   

Type 'help' to see available tools
Type 'clear' to start new chat
"""
    print(f"{Fore.BLUE}{create_centered_box(banner)}{Style.RESET_ALL}")

def chat_loop() -> None:
    """Main chat interaction loop."""
    messages: List[Dict] = [
            {"role": "system", 
            "content": "you are an Assistant in faculty of Computers and data science,"
            "you Assist students to understand the faculty rules and regulations."
            "don't make up answers, it's important to use tools every question to get information."}
        ]
    use_streaming = True  # Set to False for non-streaming mode, True for streaming

    # Clear screen on startup
    os.system('cls' if os.name == "nt" else 'clear')
    display_welcome_banner()
    show_help()

    while True:
        print(f"\n{Fore.GREEN}You{Style.RESET_ALL}: ", end="")
        user_input = input().strip()
        
        # Handle commands
        if user_input.lower() == "clear":
            messages: List[Dict] = [
                {"role": "system", 
                "content": "you are an Assistant in faculty of Computers and data science,"
                "you Assist students to understand the faculty rules and regulations."
                "don't make up answers, it's important to use tools every question to get information."}
            ]
            os.system('cls' if os.name == "nt" else 'clear')
            display_welcome_banner()
            continue
        if user_input.lower() == "help":
            show_help()
            continue

        # Process user input
        messages.append({"role": "user", "content": user_input})
        continue_tool_execution = True

        while continue_tool_execution:
            # Get response
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=Tools,
                stream=use_streaming,
                temperature=0.2
            )
            
            # Process response based on streaming mode
            if use_streaming:
                response_text, tool_calls = process_stream(response)
            else:
                response_text, tool_calls = process_non_stream(response)

            if not tool_calls:
                print()
                continue_tool_execution = False

            text_in_response = len(response_text) > 0
            if text_in_response:
                messages.append({"role": "assistant", "content": response_text})

            # Handle tool calls if any
            if tool_calls:
                tool_name = tool_calls[0]["function"]["name"]
                width = get_terminal_width()
                print(f"\n{Fore.YELLOW}[Tool Call]{Style.RESET_ALL}")
                print("─" * width)
                
                # Execute tool calls
                for tool_call in tool_calls:
                    print(f"{Fore.YELLOW}⚙ Executing{Style.RESET_ALL}: {tool_call['function']['name']}")
                    arguments = json.loads(tool_call["function"]["arguments"])
                    tool_name = tool_call["function"]["name"]

                    if tool_name == "search_info":
                        print(f"{Fore.YELLOW}• Query: {arguments['query']}{Style.RESET_ALL}")
                        results = search_info(arguments["query"], 2)
                        for i,result in enumerate(results):
                            print(f"{Fore.YELLOW}• Result({i}){Style.RESET_ALL}: {result['content']}")
                    
                    messages.append({
                            "role": "tool",
                            "content": str(results),
                            "tool_call_id": tool_call["id"]
                        })
                    print(f"{Fore.GREEN}✓ Complete{Style.RESET_ALL}")
                
                print("─" * width)

                # Continue checking for more tool calls after tool execution
                continue_tool_execution = True
            else:
                continue_tool_execution = False

if __name__ == "__main__":
    try:
        chat_loop()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
