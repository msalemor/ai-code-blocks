# Autogen

## 1 - Autogen - Overview

### 1.1 - Overview

AutoGen is an open-source programming framework designed to facilitate the creation and cooperation of AI agents. Some of its main features include:

1. **Multi-Agent Conversations**: AutoGen allows the building of next-generation LLM applications through multi-agent conversations, reducing the effort required for complex LLM workflows.
2. **Workflow Optimization**: It simplifies the orchestration, automation, and optimization of these workflows, enhancing the performance of LLM models and addressing their limitations.
3. **Diverse Conversation Patterns**: The framework supports a variety of conversation patterns, which can be crucial for complex workflows.
4. **Customizable Agents**: Developers can create agents with customizable conversation patterns, which can vary in autonomy, number of agents involved, and conversation topology.
5. **Extensive Application Range**: AutoGen includes a collection of working systems that demonstrate its ability to support diverse conversation patterns across various domains and complexities.
6. **Collaborative Research**: The framework is powered by research from Microsoft, Penn State University, and the University of Washington.

### 1.2 - Agents 

1. The `ConversableAgent` class for Agents that are capable of conversing with each other through the exchange of messages to jointly finish a task. An agent can communicate with other agents and perform actions. Different agents can differ in what actions they perform after receiving messages. Two representative subclasses are AssistantAgent and UserProxyAgent
2. The `AssistantAgent` is designed to act as an AI assistant, using LLMs by default but not requiring human input or code execution. It could write Python code (in a Python coding block) for a user to execute when a message (typically a description of a task that needs to be solved) is received. Under the hood, the Python code is written by LLM (e.g., GPT-4). It can also receive the execution results and suggest corrections or bug fixes. Its behavior can be altered by passing a new system message. The LLM inference configuration can be configured via [llm_config].
3. The `UserProxyAgent` is conceptually a proxy agent for humans, soliciting human input as the agent's reply at each interaction turn by default and also having the capability to execute code and call functions or tools. The UserProxyAgent triggers code execution automatically when it detects an executable code block in the received message and no human user input is provided. Code execution can be disabled by setting the code_execution_config parameter to False. LLM-based response is disabled by default. It can be enabled by setting llm_config to a dict corresponding to the inference configuration. When llm_config is set as a dictionary, UserProxyAgent can generate replies using an LLM when code execution is not performed.

### 1.3 - Agent LLM configuration (`llm_config`)

```python
load_dotenv()

model = os.getenv("GPT_MODEL")
endpoint=os.getenv("ENDPOINT")
api_key=os.getenv("API_KEY")
api_version=os.getenv("API_VERSION")

# An Autogen LLM can include many models and connections
# Why a list?
# - If one model times out or fails, the agent can try another model.
# - Ability to filter to a specific model
# - Specialized agents may have logic to select the best model based on the task at hand
config_list = [
    {
        "model": model,
        "base_url": endpoint,
        "api_key": api_key,
        "api_type": "azure",
        "api_version": api_version
    }
]

# An Autogen LLM configuration can include other parameters like caching, temperature and a the models and connections
llm_config = {
    "model": model,
    "temperature": 0,
    "config_list": config_list,    
    "cache_seed": None, # Disable caching (On by default)
}
```

### 1.4 - Angent creation and configuration

Suppose we want to configure a simple scenario between an agent representing a `user` and an `assistant` that can answer general questions and code generation. Suppose also that we give the agent that represents the user the ability to execute code. Let's explore the following configuration:

```python
# This agent has no LLM
# It has code execution capabilities
# It will never take human input
# It will terminate if a message is received with the word TERMINATE or if the content is empty
user = ConversableAgent(name="user",
                                max_consecutive_auto_reply=5,
                                code_execution_config={"executor": executor},
                                human_input_mode="NEVER",
                                is_termination_msg=lambda msg: "TERMINATE" in msg["content"].lower() or msg["content"]==""
                                )

# This agent has LLM capabilities
# It does not have code execution capabilities
# It will never take human input
# It will terminate if a message is received with the word TERMINATE or if the content is empty
assistant = ConversableAgent(
    name="assistant",
    code_execution_config=False,
    llm_config=llm_config,    
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"].lower() or msg["content"]=="",    
)
```
Parameters:

- `name`: The name of the agent
- `max_consecutive_auto_reply`:

### 1.5 - `initiate_chat`


```python
user.initiate_chat(assistant,message=message,clear_history=clear,silent=silent)
```

### 1.5 - Supporting classes and methods

#### Code

##### 1.5.1 - Tap into and act on conversation as it is happening

```python
chat_history = []

def process_message(
    sender: autogen.Agent,
    receiver: autogen.Agent,
    message: Dict,
    request_reply: bool = False,
    silent: bool = False,
    sender_type: str = "agent",
) -> None:
    """
    Processes the message and adds it to the agent history.

    Args:

        sender: The sender of the message.
        receiver: The receiver of the message.
        message: The message content.
        request_reply: If set to True, the message will be added to agent history.
        silent: determining verbosity.
        sender_type: The type of the sender of the message.
    """

    message = message if isinstance(message, dict) else {
        "content": message, "role": "user"}
    message_payload = {
        "recipient": receiver.name,
        "sender": sender.name,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "sender_type": sender_type,
        "message_type": "agent_message",
    }
    chat_history.append(message_payload)


class ExtendedConversableAgent(ConversableAgent):
    def __init__(self, message_processor=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_processor = message_processor

    def receive(
        self,
        message: Union[Dict, str],
        sender: autogen.Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        if self.message_processor:
            self.message_processor(sender, self, message,
                                   request_reply, silent, sender_type="agent")
        super().receive(message, sender, request_reply, silent)
```