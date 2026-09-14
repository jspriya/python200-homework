# ------------ Lesson 02: Tool Definitions and the ReAct Loop -------------

from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from scipy.stats import pearsonr
from smolagents import ToolCallingAgent, OpenAIServerModel, CodeAgent, tool

load_dotenv()
if load_dotenv():
    print("Successfully loaded environment variables from .env")
else:
    print("Warning: could not load environment variables from .env")

api_key = os.getenv("OPENAI_API_KEY")

# ------------- Q1 -------------------

def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

# JSON schema describing the function
celsius_to_fahrenheit_schema = {
    "name": "celsius_to_fahrenheit",
    "description": "Convert a Celsius temperature to Fahrenheit and return it as a formatted string.",
    "parameters": {
        "type": "object",
        "properties": {
            "celsius": {
                "type": "number",
                "description": "The temperature in degrees Celsius."
            }
        },
        "required": ["celsius"]
    }
}

# Call the function directly
print("\n" + "-" * 30 + " Q1 " + "-" * 30 )

print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))

# ------------- Q2 -------------------

# Prediction:
# 1. I predict that this query will not trigger a tool call because the model
# can calculate the Celsius-to-Fahrenheit conversion using its own internal
# knowledge. 

# 2. I predict that only one API call will be made because a second
# API call is only needed when the model requests a tool.

print("\n" + "-" * 30 + " Q2 " + "-" * 30 )

client = OpenAI()


# Define the tool function
def get_current_time() -> str:
    '''Return the current local time as a formatted string.'''
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Define the tool schema
tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Returns the current local time as a string.',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': [],
            },
        },
    }
]

def run_agent(user_prompt: str) -> str:
    '''Run a minimal ReAct-style agent for a single user prompt.'''

    SYSTEM_PROMPT = '''You are a simple assistant that can tell the current time.
                     Use the tool get_current_time whenever a user asks about the time.'''
    
    # Step 1: start the conversation with system and user messages
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    # Step 2: first API call - the model decides whether to call a tool
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',  # model chooses whether to use a tool
    )

    print("First response received from model...")
    print(first_response)
    first_message = first_response.choices[0].message

    # Record what the model said so far
    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    # Step 3: check if the model requested any tools
    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            # In this example we only have one tool: get_current_time
            if function_name == 'get_current_time':
                tool_result = get_current_time()
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            # Print for debugging so we can see what happened
            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            # Step 3b: append the tool output so the model can see it
            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        # Step 4: second API call - model sees the tool result and gives final answer
        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ''
    else:
        print("No tools needed....")

    # If there were no tool calls, the first response was already the final answer
    return first_message.content or ''

result = run_agent("Convert 100 degrees Celsius to Fahrenheit")

print(result)

# My prediction was correct. The model did not call a tool and answered
# the question directly. Only one API call was made.

# ------------------- Q3 ----------------------

# Update tools list to include both tools.

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current local time as a string.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "celsius_to_fahrenheit",
            "description": "Convert a temperature from Celsius to Fahrenheit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "celsius": {
                        "type": "number",
                        "description": "The temperature in degrees Celsius.",
                    }
                },
                "required": ["celsius"],
            },
        },
    },
]

# Update run_agent()

print("\n" + "-" * 30 + " Q3 " + "-" * 30 )

def run_agent(user_prompt: str) -> str:
    """Run a minimal ReAct-style agent for a single user prompt."""

    SYSTEM_PROMPT = """
    You are a simple assistant that can tell the current time and
    convert Celsius temperatures to Fahrenheit.
    Use the available tools when needed.
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # First API call
    first_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    print("First response received from model...")
    print(first_response)

    first_message = first_response.choices[0].message

    messages.append(
        {
            "role": "assistant",
            "content": first_message.content,
            "tool_calls": first_message.tool_calls,
        }
    )
    if first_message.tool_calls:
        print("Agentic mode engaged...")

        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name

            # Convert the JSON arguments into a Python dictionary
            function_args = json.loads(
                tool_call.function.arguments or "{}"
            )

            # Dispatch to the correct Python function
            if function_name == "get_current_time":
                tool_result = get_current_time()

            elif function_name == "celsius_to_fahrenheit":
                celsius = function_args["celsius"]
                tool_result = celsius_to_fahrenheit(celsius)

            else:
                tool_result = f"Error: unknown tool {function_name}."

            print("Tool called:", function_name)
            print("Tool arguments:", function_args)
            print("Tool result:", tool_result)

            # Add the tool result to the conversation
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result,
                }
            )

        # Second API call
        second_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
        )

        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ""

    else:
        print("No tools needed....")

    return first_message.content or ""

# Test the extended agent on both of these queries:

response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)

# A tool was called because the agent chose to use the
# celsius_to_fahrenheit tool to perform the temperature conversion.


response_b = run_agent("What is the boiling point of water in plain English?")
print("Response B:", response_b)

# No tool was called because this is a general knowledge question that
# the model can answer directly without needing the available tools.

# ------------------ Q4 ----------------------

print("\n" + "-" * 30 + " Q4 " + "-" * 30 )

class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    # --- Small internal helpers --------------------------------------

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
        return None

    # --- Tools (public methods) --------------------------------------

    def list_csv_files(self):
        """
        List available CSV files in resources/.
        """
        files = self._available_csv_files()
        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }
        return {"files": files}

    def load_csv(self, filename: str):
        """
        Load a CSV file from resources/ and make it the active dataset.

        filename can be "bike_commute" or "bike_commute.csv".
        """
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self):
        """
        Return column names for the currently loaded CSV.
        """
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """
        Return basic summary stats for one or more columns.

        If columns is None, summarize all columns.
        Uses pandas.describe(include="all") to stay simple and readable.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """
        Simple summary for a single column using pandas.describe().
        """
        error = self._ensure_loaded()
        if error:
            return error

        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """
        Plot from the active CSV.
    
        - If x is None: plot y vs row index.
        - If x is provided: plot y vs x.
        """
        error = self._ensure_loaded()
        if error:
            return error
    
        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."
    
        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"
    
        # If someone accidentally passes x == y, treat it like "plot y"
        if x == y:
            x = None
    
        # Scatter needs x
        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."
    
        title_csv = self.csv_name or "current CSV"
    
        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()
            return f"Plotted {y} vs row index as a line plot."
    
        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"
    
        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.show()
        
        return f"Plotted {y} vs {x} as a {plot_type}."
    
    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if col1 not in self.df.columns or col2 not in self.df.columns:
            return {
                "error": (
                    f"One or both columns were not found. "
                    f"Available columns: {self.df.columns.tolist()}"
                )
            }

        correlation, p_value = pearsonr(
            self.df[col1],
            self.df[col2]
        )

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(correlation, 4),
            "p_value": round(p_value, 4),
        }

print("Class defined")

# Create the CsvManager object

RESOURCES_DIR = Path("resources")
csv_backend = CsvManager(RESOURCES_DIR)

node_tools = {
    "list_csv_files": csv_backend.list_csv_files,
    "load_csv": csv_backend.load_csv,
    "get_columns": csv_backend.get_columns,
    "summarize_columns": csv_backend.summarize_columns,
    "describe_column": csv_backend.describe_column,
    "plot_data": csv_backend.plot_data,
    "compute_correlation": csv_backend.compute_correlation
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in the resources/ folder.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from the resources/ folder and make it the active dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "CSV filename in resources/, e.g. 'bike_commute.csv'.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Get the column names of the currently loaded CSV.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": "Show basic summary statistics for columns (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of column names. If omitted, summarize all columns.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Show basic summary statistics for a single column (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to describe.",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Plot data from the active CSV. If only y is provided, plot y vs row index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {"type": "string", "description": "Column name for y-axis."},
                    "x": {"type": "string", "description": "Optional column name for x-axis."},
                    "plot_type": {
                        "type": "string",
                        "enum": ["scatter", "line"],
                        "description": "Type of plot to create.",
                    },
                },
                "required": ["y"],
            },
        },
    },
    {
    "type": "function",
    "function": {
        "name": "compute_correlation",
        "description": "Compute the Pearson correlation between two columns in the loaded CSV.",
        "parameters": {
            "type": "object",
            "properties": {
                "col1": {
                    "type": "string",
                    "description": "The name of the first column."
                },
                "col2": {
                    "type": "string",
                    "description": "The name of the second column."
                }
            },
            "required": ["col1", "col2"]
        }
    }
}
]

def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """
    Run through one react-agent loop using a simple tool-using agent.
    `messages` parameter will usually just contain a system prompt, 
    and then user text will be appended.  

    The loop has three main steps:

    REASON:
      - Call the model with the conversation so far.
      - The model either replies normally, or asks to call a tool from tool set.

    ACT:
      - If tools are requested, run the Python functions

    OBSERVE:
      - Append each requested tool result back into the LLMs conversation history.
      - On the next iteration, the model reads those tool call results and determines
        whether it has reached the goal.

    Stop condition:
      - If the model returns an assistant message with no tool calls, this is the 
        final answer for this react cycle, this implies that reasoning alone without 
        tool calls was enough.  
      - max_tool_rounds is a safety cap to prevent infinite loops.
    """
    messages.append({"role": "user", "content": user_text})

    def observe_tool_result(tool_call_id, result):
        """
        Return a tool's return value as a message that can be appended to the
        LLMs conversation history. The model will read this tool output on the next
        REASON step.
        """
        content = json.dumps(result, default=str) if not isinstance(result, str) else result
        tool_message = {"role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content,}
        return tool_message

    for loop_idx in range(max_tool_rounds):
        # REASON: call the model
        # Here it will make use of any previous tool outputs it appended ("observed")
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )

        msg = response.choices[0].message

        # Append the assistant message to the conversation history.
        # Use a plain dict so `messages` stays simple and inspectable.
        assistant_entry = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_entry)

        # No tool calls means the model is answering directly.
        if not msg.tool_calls:
            return msg.content 

        # ACT + OBSERVE: run each tool call, then append its result.
        # Note there may be multiple tool calls
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")

            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)
            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try:
                    result = fn(**tool_args) if tool_args else fn()
                except Exception as e:
                    print(f"Tool error in {name}: {type(e).__name__}: {e}")
                    result = {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}
                    
            # OBSERVE: append the tool result back into the conversation history.
            messages.append(observe_tool_result(tool_call.id, result))
            
            # After we appending information about all tool outputs, we loop back and REASON again.

    return "I hit the tool-round limit. Try a simpler request."


# Test: Compute correlation between two columns
"""
# Used the below code to test Q4 setup

messages = [
    {
        "role": "system",
        "content": (
            "You are a data analysis assistant. "
            "Use the available tools to inspect CSV files and answer questions about the data."
        ),
    }
]

result = run_agent_cycle(
    messages,
    "Compute the correlation between any two numeric columns."
)
"""

# Q5 testing
print("\n" + '-'*35 + " Q5 " + '-'*35)

messages = [{"role": "system", "content": (
            "You are a data analysis assistant. "
            "Use the available tools to inspect CSV files and answer questions about the data."
        ),}]

result = run_agent_cycle(
    messages,
    "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh."
)
print(result)

# -------------- Q6 ----------------

print("\n" + '-'*35 + " Q6 " + '-'*35)
print(json.dumps(messages, indent=2, default=str))

# --------------- Q7 - smolagents ------------------

print("\n" + '-'*35 + " Q7 " + '-'*35)

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """
    Compute the Pearson correlation between two columns in the currently loaded CSV.

    Args:
        col1: The name of the first numeric column.
        col2: The name of the second numeric column.

    Returns:
        A dictionary containing the column names, Pearson correlation coefficient,
        and p-value, or an error message if the calculation cannot be performed.
    """
    return csv_backend.compute_correlation(col1, col2)

print(compute_correlation.description)

# In Q4, we manually created a JSON schema that specified the tool's name,
# description, parameters, parameter types, and required fields. With smolagents,
# this information is generated automatically from the function definition.
# To generate a good description, smolagents needs the developer to provide a
# clear function name, type hints for the parameters and return value, and a
# descriptive docstring explaining what the tool does and what each argument means.

# --------------- Q8 - ToolCallingAgent vs CodeAgent ------------------
csv_manager = csv_backend

print("\n" + "-" * 35 + " Q8 " + "-" * 35)

@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.

    Returns:
        A dict with a "files" list, or a message if none are found.
    """
    return csv_manager.list_csv_files()


@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.

    Args:
        filename: CSV filename in resources/. You can pass "bike_commute" or "bike_commute.csv".

    Returns:
        A dict with a status message and column names, or an error dict.
    """
    return csv_manager.load_csv(filename)


@tool
def get_columns() -> list[str] | dict:
    """Return column names for the currently loaded CSV.

    Returns:
        A list of column names, or an error dict if no CSV is loaded.
    """
    return csv_manager.get_columns()


@tool
def summarize_columns(columns: list[str] | None = None) -> dict:
    """Return summary stats for selected columns (or all columns). 
    This includes count, mean, std, min, max, and percentiles for numeric columns,
    or count, unique, top, freq for categorical columns.

    Args:
        columns: Column names to summarize. If None, summarizes all columns.

    Returns:
        A dict of summary statistics (from pandas.describe), or an error dict.
    """
    return csv_manager.summarize_columns(columns)


@tool
def describe_column(column: str) -> dict:
    """Describe a single column (basic stats) for the requested column.
    This includes count, mean, std, min, max, and percentiles for numeric column,
    or count, unique, top, freq for categorical column.

    Args:
        column: The name of the column to describe.

    Returns:
        A dict of basic stats for the column, or an error dict.
    """
    return csv_manager.describe_column(column)


@tool
def plot_data(y: str, x: str | None = None, plot_type: str = "line") -> str | dict:
    """Plot from the active CSV.

    Args:
        y: Column name to plot on the y-axis. 
        x: Column name to plot on the x-axis. If None, use row index.
        plot_type: "line" or "scatter". Scatter requires x and y.

    Returns:
        Generates and shows the plot. 
        Retirms a short success message string, or an error dict/string.
    """
    return csv_manager.plot_data(y=y, x=x, plot_type=plot_type)

# Include the new Q7 tool
TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation,
]

model_to_use = "gpt-4o-mini"

model = OpenAIServerModel(
    api_key=api_key,
    model_id=model_to_use,
)

SYSTEM_PROMPT = (
    "You are a small data assistant to help analyze files stored in resources/. "
    "Use the available tools to do any work requested (do not guess). "
    "Keep answers short and student-friendly."
)

tool_agent = ToolCallingAgent(
    tools=TOOLS,
    model=model,
    instructions=SYSTEM_PROMPT,
)

code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions=SYSTEM_PROMPT,
)

prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."

response_tool = tool_agent.run(prompt)

response_code = code_agent.run(
    prompt,
    additional_args={"csv_manager": csv_backend}
)

print("\nToolCallingAgent response:")
print(response_tool)

print("\nCodeAgent response:")
print(response_code)

# observations:
#
# The ToolCallingAgent successfully loaded bike_commute.csv and called
# plot_data() with avg_heart_rate as y, duration_min as x, and scatter
# as the plot type. However, it did not change the dot color to green.
# The plot used the default color because the plot_data tool does not
# have a color parameter.
#
# The CodeAgent also attempted to load the CSV and call plot_data() with
# the same arguments. However, its execution encountered a Matplotlib
# MacOS GUI/backend error because the CodeAgent runs generated code in
# a separate execution environment. It then tried to change the
# Matplotlib backend, but the import was not authorized. Therefore,
# the CodeAgent did not successfully complete the plotting task.
#
# This shows that ToolCallingAgent is useful when we want the model to
# use predefined tools in a controlled way. The agent can only do what
# the available tools support. CodeAgent is more flexible because it can
# write and execute Python code, which can be useful for more complex
# analysis. However, CodeAgent also depends on its code execution
# environment and authorized imports, so it can encounter execution
# restrictions that a ToolCallingAgent may avoid.

# ------------------ Q9 ------------------
#
# 1. A ToolCallingAgent would be a better choice for a task such as
# loading a CSV file and calculating a specific statistic, such as
# the correlation between two columns. This is a good fit because
# the task can be completed using predefined tools with known inputs
# and outputs. The agent is limited to the available tools, which
# makes the process more controlled and predictable.
#
# 2. One meaningful risk of using a CodeAgent is that the agent
# generates and executes Python code. The generated code could make
# unintended changes, perform unsafe operations, or fail because of
# the execution environment or unauthorized imports. A ToolCallingAgent
# is more controlled because it can only call the specific tools that
# have been provided to it.