"""
LangChain Agent Implementation

This module demonstrates how to build AI agents using LangChain framework.
Includes examples of different agent types, tool usage, and memory systems.
"""

import os
from typing import List, Dict, Any, Optional, Callable
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools import BaseTool, DuckDuckGoSearchRun, ShellTool
from langchain.tools.python.tool import PythonREPLTool
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.llms.base import LLM
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import requests
import json
import sqlite3
from datetime import datetime
import numpy as np


class CustomTool(BaseTool):
    """Custom tool template for creating specialized agent tools"""
    
    name: str = "custom_tool"
    description: str = "A custom tool for specific tasks"
    
    def _run(self, query: str) -> str:
        """Execute the tool with the given query"""
        # Implement your custom logic here
        return f"Custom tool executed with query: {query}"
    
    async def _arun(self, query: str) -> str:
        """Async version of the tool execution"""
        return self._run(query)


class WeatherTool(BaseTool):
    """Tool for getting weather information"""
    
    name: str = "weather"
    description: str = "Get current weather information for a given location"
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or os.getenv("WEATHER_API_KEY")
    
    def _run(self, location: str) -> str:
        """Get weather for the specified location"""
        if not self.api_key:
            return "Weather API key not configured"
        
        try:
            # Example using OpenWeatherMap API
            url = f"http://api.openweathermap.org/data/2.5/weather"
            params = {
                "q": location,
                "appid": self.api_key,
                "units": "metric"
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if response.status_code == 200:
                weather = data["weather"][0]["description"]
                temp = data["main"]["temp"]
                humidity = data["main"]["humidity"]
                
                return f"Weather in {location}: {weather}, Temperature: {temp}°C, Humidity: {humidity}%"
            else:
                return f"Could not get weather for {location}: {data.get('message', 'Unknown error')}"
        
        except Exception as e:
            return f"Error getting weather: {str(e)}"
    
    async def _arun(self, location: str) -> str:
        return self._run(location)


class DatabaseTool(BaseTool):
    """Tool for interacting with SQLite databases"""
    
    name: str = "database"
    description: str = "Execute SQL queries on the database"
    
    def __init__(self, db_path: str = "agent_memory.db"):
        super().__init__()
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the database with basic tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create a simple notes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                content TEXT,
                tags TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _run(self, query: str) -> str:
        """Execute SQL query"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Simple safety check - only allow SELECT, INSERT, UPDATE
            query_type = query.strip().upper().split()[0]
            if query_type not in ["SELECT", "INSERT", "UPDATE"]:
                return "Only SELECT, INSERT, and UPDATE queries are allowed"
            
            cursor.execute(query)
            
            if query_type == "SELECT":
                results = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                if results:
                    formatted_results = []
                    for row in results:
                        row_dict = dict(zip(columns, row))
                        formatted_results.append(row_dict)
                    return json.dumps(formatted_results, indent=2)
                else:
                    return "No results found"
            else:
                conn.commit()
                return f"Query executed successfully. Rows affected: {cursor.rowcount}"
        
        except Exception as e:
            return f"Database error: {str(e)}"
        finally:
            conn.close()
    
    async def _arun(self, query: str) -> str:
        return self._run(query)


class CalculatorTool(BaseTool):
    """Tool for mathematical calculations"""
    
    name: str = "calculator"
    description: str = "Perform mathematical calculations. Input should be a mathematical expression."
    
    def _run(self, expression: str) -> str:
        """Evaluate mathematical expression safely"""
        try:
            # Simple safety check - only allow basic math operations
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                return "Invalid characters in expression. Only numbers and basic operators allowed."
            
            # Use eval with restricted globals for safety
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        
        except Exception as e:
            return f"Calculation error: {str(e)}"
    
    async def _arun(self, expression: str) -> str:
        return self._run(expression)


class AgentCallbackHandler(BaseCallbackHandler):
    """Custom callback handler to monitor agent behavior"""
    
    def __init__(self):
        self.steps = []
        self.thoughts = []
    
    def on_agent_action(self, action, **kwargs):
        """Called when agent takes an action"""
        self.steps.append({
            "action": action.tool,
            "input": action.tool_input,
            "log": action.log
        })
        print(f"🤖 Agent Action: {action.tool}")
        print(f"📝 Input: {action.tool_input}")
    
    def on_agent_finish(self, finish, **kwargs):
        """Called when agent finishes"""
        print(f"✅ Agent Finished: {finish.return_values}")
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Called when tool starts"""
        print(f"🔧 Tool Start: {serialized.get('name', 'Unknown')}")
    
    def on_tool_end(self, output, **kwargs):
        """Called when tool ends"""
        print(f"🔧 Tool Output: {output}")
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called when LLM starts"""
        print(f"🧠 LLM Thinking...")
    
    def on_llm_end(self, response, **kwargs):
        """Called when LLM ends"""
        if hasattr(response, 'generations') and response.generations:
            text = response.generations[0][0].text
            self.thoughts.append(text)


class ReActAgent:
    """
    ReAct (Reasoning + Acting) Agent implementation
    
    This agent follows the ReAct pattern:
    1. Thought: Reason about the current situation
    2. Action: Take an action using available tools
    3. Observation: Observe the result of the action
    4. Repeat until task is complete
    """
    
    def __init__(
        self,
        llm: LLM,
        tools: List[BaseTool],
        memory: Optional[Any] = None,
        max_iterations: int = 10,
        verbose: bool = True
    ):
        self.llm = llm
        self.tools = tools
        self.memory = memory or ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Create the agent
        self.agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            max_iterations=max_iterations,
            verbose=verbose,
            handle_parsing_errors=True
        )
        
        # Add callback handler
        self.callback_handler = AgentCallbackHandler()
        self.agent.callback_manager.add_handler(self.callback_handler)
    
    def run(self, input_text: str) -> str:
        """Run the agent with the given input"""
        try:
            result = self.agent.run(input_text)
            return result
        except Exception as e:
            return f"Agent error: {str(e)}"
    
    def get_conversation_history(self) -> List[BaseMessage]:
        """Get the conversation history"""
        if hasattr(self.memory, 'chat_memory'):
            return self.memory.chat_memory.messages
        return []
    
    def clear_memory(self):
        """Clear the agent's memory"""
        self.memory.clear()
    
    def get_agent_steps(self) -> List[Dict]:
        """Get the steps taken by the agent"""
        return self.callback_handler.steps
    
    def get_agent_thoughts(self) -> List[str]:
        """Get the agent's thoughts/reasoning"""
        return self.callback_handler.thoughts


class TaskPlanningAgent:
    """
    Agent that can break down complex tasks into subtasks
    and execute them systematically
    """
    
    def __init__(self, llm: LLM, tools: List[BaseTool]):
        self.llm = llm
        self.tools = tools
        
        # Task planning prompt
        self.planning_prompt = PromptTemplate(
            input_variables=["task", "available_tools"],
            template="""
            You are a task planning agent. Break down the following task into smaller, manageable subtasks.
            
            Task: {task}
            
            Available tools: {available_tools}
            
            Create a step-by-step plan where each step can be executed using the available tools.
            Format your response as a numbered list of subtasks.
            
            Plan:
            """
        )
        
        self.planning_chain = LLMChain(llm=llm, prompt=self.planning_prompt)
        
        # Execution agent
        self.executor = ReActAgent(llm, tools, verbose=False)
    
    def plan_task(self, task: str) -> List[str]:
        """Create a plan for the given task"""
        tool_descriptions = [f"{tool.name}: {tool.description}" for tool in self.tools]
        available_tools = "\n".join(tool_descriptions)
        
        plan_text = self.planning_chain.run(
            task=task,
            available_tools=available_tools
        )
        
        # Parse the plan into individual steps
        steps = []
        for line in plan_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering and clean up
                step = line.split('.', 1)[-1].strip()
                if step.startswith('-'):
                    step = step[1:].strip()
                steps.append(step)
        
        return steps
    
    def execute_plan(self, task: str) -> Dict[str, Any]:
        """Plan and execute a complex task"""
        print(f"📋 Planning task: {task}")
        
        # Create plan
        steps = self.plan_task(task)
        
        print(f"📝 Plan created with {len(steps)} steps:")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")
        
        # Execute each step
        results = []
        for i, step in enumerate(steps, 1):
            print(f"\n🚀 Executing step {i}: {step}")
            
            try:
                result = self.executor.run(step)
                results.append({
                    "step": i,
                    "description": step,
                    "result": result,
                    "status": "success"
                })
                print(f"✅ Step {i} completed")
            
            except Exception as e:
                results.append({
                    "step": i,
                    "description": step,
                    "result": str(e),
                    "status": "error"
                })
                print(f"❌ Step {i} failed: {e}")
        
        return {
            "task": task,
            "plan": steps,
            "results": results,
            "success": all(r["status"] == "success" for r in results)
        }


class MultiAgentSystem:
    """
    System for coordinating multiple specialized agents
    """
    
    def __init__(self, llm: LLM):
        self.llm = llm
        self.agents = {}
        self.coordinator_prompt = PromptTemplate(
            input_variables=["task", "available_agents"],
            template="""
            You are a coordinator for a multi-agent system. Given the following task and available agents,
            determine which agent(s) should handle this task and in what order.
            
            Task: {task}
            
            Available agents: {available_agents}
            
            Respond with the name of the agent that should handle this task, or "MULTIPLE" if multiple agents are needed.
            If multiple agents are needed, specify the order and coordination strategy.
            
            Decision:
            """
        )
        
        self.coordinator = LLMChain(llm=llm, prompt=self.coordinator_prompt)
    
    def add_agent(self, name: str, agent: ReActAgent, description: str):
        """Add a specialized agent to the system"""
        self.agents[name] = {
            "agent": agent,
            "description": description
        }
    
    def route_task(self, task: str) -> str:
        """Determine which agent should handle the task"""
        agent_descriptions = [
            f"{name}: {info['description']}" 
            for name, info in self.agents.items()
        ]
        available_agents = "\n".join(agent_descriptions)
        
        decision = self.coordinator.run(
            task=task,
            available_agents=available_agents
        )
        
        return decision.strip()
    
    def execute_task(self, task: str) -> Dict[str, Any]:
        """Route and execute a task using the appropriate agent(s)"""
        routing_decision = self.route_task(task)
        
        print(f"🎯 Routing decision: {routing_decision}")
        
        # Simple routing - use first mentioned agent name
        selected_agent = None
        for agent_name in self.agents.keys():
            if agent_name.lower() in routing_decision.lower():
                selected_agent = agent_name
                break
        
        if selected_agent:
            print(f"🤖 Using agent: {selected_agent}")
            agent = self.agents[selected_agent]["agent"]
            result = agent.run(task)
            
            return {
                "task": task,
                "selected_agent": selected_agent,
                "routing_decision": routing_decision,
                "result": result
            }
        else:
            return {
                "task": task,
                "selected_agent": None,
                "routing_decision": routing_decision,
                "result": "No suitable agent found for this task"
            }


# Example usage and testing
if __name__ == "__main__":
    # Note: You'll need to set up your LLM (e.g., OpenAI API key)
    # For this example, we'll create a mock LLM
    
    class MockLLM(LLM):
        """Mock LLM for testing purposes"""
        
        def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
            # Simple mock responses based on prompt content
            if "weather" in prompt.lower():
                return "I need to check the weather using the weather tool."
            elif "calculate" in prompt.lower():
                return "I need to perform a calculation using the calculator tool."
            elif "plan" in prompt.lower():
                return "1. First step\n2. Second step\n3. Third step"
            else:
                return "I'll help you with that task."
        
        @property
        def _llm_type(self) -> str:
            return "mock"
    
    # Create tools
    tools = [
        WeatherTool(),
        CalculatorTool(),
        DatabaseTool(),
        DuckDuckGoSearchRun(),
    ]
    
    # Create LLM
    llm = MockLLM()
    
    # Test ReAct Agent
    print("=== Testing ReAct Agent ===")
    agent = ReActAgent(llm, tools, verbose=True)
    
    # Test simple query
    result = agent.run("What's 25 * 4?")
    print(f"Result: {result}")
    
    # Test Task Planning Agent
    print("\n=== Testing Task Planning Agent ===")
    planner = TaskPlanningAgent(llm, tools)
    
    task_result = planner.execute_plan("Calculate the area of a circle with radius 5")
    print(f"Task execution result: {task_result}")
    
    # Test Multi-Agent System
    print("\n=== Testing Multi-Agent System ===")
    multi_agent = MultiAgentSystem(llm)
    
    # Create specialized agents
    math_agent = ReActAgent(llm, [CalculatorTool()], verbose=False)
    weather_agent = ReActAgent(llm, [WeatherTool()], verbose=False)
    
    multi_agent.add_agent("math", math_agent, "Handles mathematical calculations")
    multi_agent.add_agent("weather", weather_agent, "Provides weather information")
    
    # Test task routing
    routing_result = multi_agent.execute_task("What's the square root of 144?")
    print(f"Multi-agent result: {routing_result}")
    
    print("\n✅ All agent tests completed!")