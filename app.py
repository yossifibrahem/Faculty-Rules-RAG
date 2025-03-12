import os
import uuid
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, send_from_directory, session
import json
from datetime import datetime
from openai import OpenAI
import time

# Local imports
from tool_managment import RAG
from tool_managment import FAQ
from tool_managment import links


app = Flask(__name__)
app.static_folder="templates"

# Configure session
app.secret_key = 'JO12345678!_'  # Replace with a real secret key
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(os.path.expanduser('~'), '.flask_sessions')
app.config['PERMANENT_SESSION_LIFETIME'] = 31536000  # 1 year in seconds

# Create session directory if it doesn't exist
if not os.path.exists(app.config['SESSION_FILE_DIR']):
    os.makedirs(app.config['SESSION_FILE_DIR'])

@app.route('/style.css')
def serve_css():
    return send_from_directory('templates', 'style.css')

client = OpenAI(
    base_url=os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
    api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio")
)
model_name = os.getenv("LMSTUDIO_MODEL", "lmstudio-community/qwen2.5-7b-instruct")
MODEL = model_name

@app.route('/')
def home():
    return render_template('index.html')

# Tool definitions
Tools = [{
    "type": "function",
    "function": {
        "name": "RAG",
        "description": (
            "This tool Retrieve a snipped article from faculty rules and regulations based on the query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "general query to search for"}
            },
            "required": ["query"]
        }
    }
}, {
    "type": "function",
    "function": {
        "name": "FAQ",
        "description": (
            "This tool Retrieve a relevent frequently asked questions and answers based on the query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "general query to search for"},
                "top_k": {"type": "integer", "description": "number of results to return"}
            },
            "required": ["query"]
        }
    }
}, {
    "type": "function",
    "function": {
        "name": "links",
        "description": (
            "This tool provides links to useful resources."
            "use it if the student asks for a link to a specific resource."
        ),
    }
}]

CONVERSATIONS_DIR = os.path.expanduser("~/.conversations")
if not os.path.exists(CONVERSATIONS_DIR):
    os.makedirs(CONVERSATIONS_DIR)

current_conversation_id = None

# Modified structure to store user-specific chat messages
user_sessions = {}

# Modified conversation directory structure
def get_user_conversation_dir(user_id):
    user_dir = os.path.join(CONVERSATIONS_DIR, user_id)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    return user_dir

@app.before_request
def before_request():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        session.permanent = True  # Make the session permanent
    
    # Initialize user session if needed
    user_id = session['user_id']
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            'current_conversation_id': None,
            'chat_messages': [
                {"role": "system", 
                "content": "you are an Assistant in faculty of Computers and data science,"
                "you Assist students."
                "don't make up answers, it's important to use tools every question to get information."}
            ]
        }

def save_conversation():
    user_id = session['user_id']
    user_data = user_sessions[user_id]
    if user_data['current_conversation_id']:
        conversation_data = {
            "id": user_data['current_conversation_id'],
            "last_updated": datetime.now().isoformat(),
            "messages": user_data['chat_messages'],
            "name": get_conversation_name(user_data['chat_messages']) if user_data['chat_messages'] else "New Conversation"
        }
        user_dir = get_user_conversation_dir(user_id)
        with open(f"{user_dir}/{user_data['current_conversation_id']}.json", "w") as f:
            json.dump(conversation_data, f, indent=2)

def load_conversation(conversation_id):
    user_id = session['user_id']
    try:
        user_dir = get_user_conversation_dir(user_id)
        with open(f"{user_dir}/{conversation_id}.json", "r") as f:
            data = json.load(f)
            return data["messages"]
    except:
        return []

def get_all_conversations():
    user_id = session['user_id']
    user_dir = get_user_conversation_dir(user_id)
    conversations = []
    if os.path.exists(user_dir):
        for filename in os.listdir(user_dir):
            if filename.endswith(".json"):
                with open(f"{user_dir}/{filename}", "r") as f:
                    data = json.load(f)
                    conversations.append({
                        "id": data["id"],
                        "last_updated": data["last_updated"],
                        "name": data.get("name", "Unnamed Conversation"),
                        "preview": data["messages"][0]["content"] if data["messages"] else "Empty conversation"
                    })
    return sorted(conversations, key=lambda x: x["last_updated"], reverse=True)

def get_conversation_name(messages):
    user_id = session['user_id']
    user_data = user_sessions[user_id]
    current_conv_id = user_data['current_conversation_id']
    user_dir = get_user_conversation_dir(user_id)

    user_messages = [msg for msg in messages if msg["role"] == "user"]
    assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]

    if len(user_messages) > 2:
        try:
            with open(f"{user_dir}/{current_conv_id}.json", "r") as f:
                data = json.load(f)
                return data["name"]
        except:
            pass  # If file doesn't exist yet, generate new name

    number_of_messages = len(user_messages) + len(assistant_messages)
    conv = json.dumps(messages[:number_of_messages] if number_of_messages else messages)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant specializing in creating concise conversation titles. "
                        "Create a brief, relevant title (maximum 25 characters) for this conversation "
                        "based on the 1-3 user messages and assistant responses. "
                        "Return only the title, no quotes or extra text."
                    )
                },
                {
                    "role": "user",
                    "content": (conv)
                }
            ],
            temperature = 1
        )
        return response.choices[0].message.content.strip()[:40]
    except Exception as e:
        print(f"Error generating conversation name: {e}")
        return "New Conversation"

@app.route('/chat', methods=['POST'])
def chat():
    user_id = session['user_id']
    user_data = user_sessions[user_id]
    
    user_message = request.json.get('message')
    stream = request.json.get('stream', False)
    user_data['chat_messages'].append({"role": "user", "content": str(user_message)})
    
    if not user_data['current_conversation_id']:
        user_data['current_conversation_id'] = str(uuid.uuid4())
    
    def generate_response():
        # Replace global references with user_data references
        chat_messages = user_data['chat_messages']
        continue_tool_execution = True
        
        while continue_tool_execution:
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=chat_messages,
                    tools=Tools,
                    stream=stream,
                    temperature=0.2
                )
                
                if not stream:
                    # Handle non-streaming response
                    if response.choices[0].message.content:
                        chat_messages.append({"role": "assistant", "content": response.choices[0].message.content})
                        yield f"data: {json.dumps({'type': 'content', 'content': response.choices[0].message.content})}\n\n"
                    
                    if response.choices[0].message.tool_calls:
                        # Convert tool calls to dict format for storage
                        tool_calls = [{
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in response.choices[0].message.tool_calls]
                        
                        chat_messages.append({"role": "assistant", "tool_calls": tool_calls})
                        
                        for tool_call in tool_calls:
                            arguments = json.loads(tool_call["function"]["arguments"])
                            tool_name = tool_call["function"]["name"]

                            yield f"data: {json.dumps({'type': 'tool-start', 'name': tool_name, 'args': arguments})}\n\n"
                            
                            if tool_name == "RAG":
                                result = RAG(arguments["query"], 3)
                            elif tool_name == "FAQ":
                                result = FAQ(arguments["query"], arguments.get("top_k", 1))
                            elif tool_name == "links":
                                result = links()
                                
                            chat_messages.append({
                                "role": "tool",
                                "content": str(result),
                                "tool_call_id": tool_call["id"]
                            })
                            yield f"data: {json.dumps({'type': 'tool', 'name': tool_name, 'content': result, 'args': arguments})}\n\n"
                        continue_tool_execution = True
                    else:
                        continue_tool_execution = False
                else:
                    # Existing streaming code
                    current_message = ""
                    tool_calls = []
                    
                    for chunk in response:
                        if interrupt_flag:
                            continue_tool_execution = False
                            break

                        delta = chunk.choices[0].delta
                        
                        if delta.content is not None:
                            current_message += delta.content
                            yield f"data: {json.dumps({'type': 'content', 'content': delta.content})}\n\n"
                        
                        elif delta.tool_calls:
                            for tc in delta.tool_calls:
                                # Show indicator when Python tool is called
                                if tc.function and tc.function.name == "python": yield f"data: {json.dumps({'type': 'tool-start', 'name': 'coding', 'args': {'code': 'Writing code...'}})}\n\n"
                                
                                if len(tool_calls) <= tc.index:
                                    tool_calls.append({
                                        "id": "", "type": "function",
                                        "function": {"name": "", "arguments": ""}
                                    })
                                tool_calls[tc.index] = {
                                    "id": (tool_calls[tc.index]["id"] + (tc.id or "")),
                                    "type": "function",
                                    "function": {
                                        "name": (tool_calls[tc.index]["function"]["name"] + (tc.function.name or "")),
                                        "arguments": (tool_calls[tc.index]["function"]["arguments"] + (tc.function.arguments or ""))
                                    }
                                }
                    
                    if current_message:
                        chat_messages.append({"role": "assistant", "content": current_message})
                    
                    if tool_calls:
                        # Convert tool calls to dict format
                        formatted_tool_calls = [{
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"]
                            }
                        } for tc in tool_calls]
                        
                        chat_messages.append({"role": "assistant", "tool_calls": formatted_tool_calls})
                        
                        for tool_call in tool_calls:
                            arguments = json.loads(tool_call["function"]["arguments"])
                            tool_name = tool_call["function"]["name"]

                            yield f"data: {json.dumps({'type': 'tool-start', 'name': tool_name, 'args': arguments})}\n\n"
                            
                            if tool_name == "RAG":
                                result = RAG(arguments["query"], 3)

                            elif tool_name == "FAQ":
                                result = FAQ(arguments["query"], arguments.get("top_k", 1))

                            elif tool_name == "links":
                                result = links()
                                
                            chat_messages.append({
                                        "role": "tool",
                                        "content": str(result),
                                        "tool_call_id": tool_call["id"]
                                    })
                            yield f"data: {json.dumps({'type': 'tool', 'name': tool_name, 'content': result, 'args': arguments})}\n\n"
                        
                        continue_tool_execution = True
                    else:
                        continue_tool_execution = False
                    
            except Exception as e:
                print(f"Error in generate_response: {e}")
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
                continue_tool_execution = False
                
        yield "data: [DONE]\n\n"
        save_conversation()

    return Response(stream_with_context(generate_response()), mimetype='text/event-stream')

@app.route('/conversations', methods=['GET'])
def list_conversations():
    return jsonify(get_all_conversations())

@app.route('/conversation/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    user_id = session['user_id']
    user_data = user_sessions[user_id]
    user_data['current_conversation_id'] = conversation_id
    user_data['chat_messages'] = load_conversation(conversation_id)
    return jsonify({"status": "success", "messages": user_data['chat_messages']})

@app.route('/new', methods=['POST'])
def new_conversation():
    user_id = session['user_id']
    user_data = user_sessions[user_id]
    user_data['current_conversation_id'] = str(uuid.uuid4())
    user_data['chat_messages'] = [
        {"role": "system", 
        "content": "you are an Assistant in faculty of Computers and data science,"
        "you Assist students."
        "don't make up answers, it's important to use tools every question to get information."}
    ]
    return jsonify({
        "status": "success", 
        "conversation_id": user_data['current_conversation_id'],
        "name": "New Conversation"
    })

@app.route('/conversation/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    user_id = session['user_id']
    user_data = user_sessions[user_id]
    try:
        user_dir = get_user_conversation_dir(user_id)
        file_path = f"{user_dir}/{conversation_id}.json"
        if os.path.exists(file_path):
            os.remove(file_path)
            if user_data['current_conversation_id'] == conversation_id:
                user_data['current_conversation_id'] = None
                user_data['chat_messages'] = [
                    {"role": "system", 
                    "content": "you are an Assistant in faculty of Computers and data science,"
                    "you Assist students."
                    "don't make up answers, it's important to use tools every question to get information."}
                ]
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "error", "message": "Conversation not found"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/interrupt', methods=['POST'])
def interrupt():
    global client, interrupt_flag
    interrupt_flag = True
    time.sleep(0.1)
    client.close()
    client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
    interrupt_flag = False
    return jsonify({"status": "success"})

@app.route('/messages', methods=['GET'])
def get_messages():
    user_id = session['user_id']
    user_data = user_sessions[user_id]
    chat_messages = user_data['chat_messages']
    formatted_messages = []
    current_tool_results = []
    current_tool_args = {}
    
    for msg in chat_messages:
        if msg["role"] == "user":
            formatted_messages.append({
                "isUser": True,
                "content": msg["content"]
            })
        elif msg["role"] == "assistant":
            if "tool_calls" in msg:
                # Convert tool_calls objects to dict format for JSON serialization
                if isinstance(msg["tool_calls"], list):
                    current_tool_results = [
                        {
                            "id": tc.id if hasattr(tc, 'id') else tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc.function.name if hasattr(tc, 'function') else tc["function"]["name"],
                                "arguments": tc.function.arguments if hasattr(tc, 'function') else tc["function"]["arguments"]
                            }
                        } for tc in msg["tool_calls"]
                    ]
                else:
                    current_tool_results = msg["tool_calls"]
                
                current_tool_args = {
                    tc["id"]: json.loads(tc["function"]["arguments"])
                    for tc in current_tool_results
                }
                
            formatted_messages.append({
                "isUser": False,
                "content": msg.get("content", ""),
                "tool_calls": current_tool_results
            })
        elif msg["role"] == "tool":
            if formatted_messages and not formatted_messages[-1]["isUser"]:
                if "tool_results" not in formatted_messages[-1]:
                    formatted_messages[-1]["tool_results"] = []
                
                tool_call = next((tc for tc in current_tool_results 
                               if tc["id"] == msg["tool_call_id"]), None)
                
                if tool_call:
                    tool_name = tool_call["function"]["name"]
                    try:
                        content = eval(msg["content"])
                    except:
                        content = msg["content"]
                    
                    formatted_messages[-1]["tool_results"].append({
                        "name": tool_name,
                        "content": content,
                        "args": current_tool_args.get(msg["tool_call_id"], {})
                    })
    
    return jsonify(formatted_messages)

@app.route('/delete-last', methods=['POST'])
def delete_last_message():
    user_id = session['user_id']
    user_data = user_sessions[user_id]
    chat_messages = user_data['chat_messages']
    if len(chat_messages) >= 2:
        # Find the last user message index
        last_user_index = None
        for i in range(len(chat_messages) - 1, -1, -1):
            if chat_messages[i]["role"] == "user":
                last_user_index = i
                break
        
        if last_user_index is not None:
            # Remove all messages after and including the last user message
            user_data['chat_messages'] = chat_messages[:last_user_index]
            save_conversation()
            return jsonify({"status": "success"})
            
    return jsonify({"status": "error", "message": "No messages to delete"}), 400

@app.route('/regenerate', methods=['POST'])
def regenerate_response():
    user_id = session['user_id']
    user_data = user_sessions[user_id]
    chat_messages = user_data['chat_messages']
    if len(chat_messages) >= 1:
        last_user_message = None
        messages_to_remove = 0
        
        for msg in reversed(chat_messages):
            messages_to_remove += 1
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break
        
        if last_user_message:
            user_data['chat_messages'] = chat_messages[:-messages_to_remove]
            save_conversation()
            return jsonify({"status": "success", "message": last_user_message})
            
    return jsonify({"status": "error", "message": "No message to regenerate"}), 400

if __name__ == '__main__':
    app.run(debug=True)