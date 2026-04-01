from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic


# ── LLM Setup ────────────────────────────────────────────────────────────────

llm = ChatAnthropic(model="claude-opus-4-5")


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def generate_mcq(input: str) -> str:
    """Generate multiple choice questions from study notes.
    Input format: 'notes: <your notes> | num_questions: <number>'
    Generates MCQs with 4 options and one correct answer."""
    parts = input.split("|")
    notes = parts[0].replace("notes:", "").strip()
    num = 3
    if len(parts) > 1:
        try:
            num = int(parts[1].replace("num_questions:", "").strip())
        except ValueError:
            pass

    prompt = f"""Generate {num} multiple-choice questions based on these notes.

Notes:
{notes}

Format each question exactly like this:
Q1: <question>
A) <option>
B) <option>
C) <option>
D) <option>
Answer: <letter>

---"""
    response = llm.invoke(prompt)
    return response.content


@tool
def generate_true_false(input: str) -> str:
    """Generate true/false questions from study notes.
    Input format: 'notes: <your notes> | num_questions: <number>'
    Generates true/false statements with explanations."""
    parts = input.split("|")
    notes = parts[0].replace("notes:", "").strip()
    num = 3
    if len(parts) > 1:
        try:
            num = int(parts[1].replace("num_questions:", "").strip())
        except ValueError:
            pass

    prompt = f"""Generate {num} true/false questions based on these notes.

Notes:
{notes}

Format each question exactly like this:
Q1: <statement>
Answer: True / False
Explanation: <one sentence why>

---"""
    response = llm.invoke(prompt)
    return response.content


@tool
def generate_flashcards(input: str) -> str:
    """Generate flashcards from study notes.
    Input format: 'notes: <your notes> | num_cards: <number>'
    Generates front/back flashcard pairs for key concepts."""
    parts = input.split("|")
    notes = parts[0].replace("notes:", "").strip()
    num = 4
    if len(parts) > 1:
        try:
            num = int(parts[1].replace("num_cards:", "").strip())
        except ValueError:
            pass

    prompt = f"""Generate {num} flashcards based on these notes.

Notes:
{notes}

Format each flashcard exactly like this:
Card 1:
  Front: <question or term>
  Back:  <answer or definition>

---"""
    response = llm.invoke(prompt)
    return response.content


# ── ReAct Prompt ──────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """You are a study assistant agent that helps students learn by generating study materials from their notes.

You have access to the following tools:
{tools}

Use the following format strictly:

Question: the input question you must answer
Thought: think about what study materials to generate and which tools to use
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have all the study materials ready
Final Answer: present all the generated study materials in a clear, well-formatted way

Instructions:
- Always use the tools to generate study materials — never write questions yourself
- Pass the full notes in the Action Input using the format: "notes: <notes> | num_questions: <n>"
- If the user asks for multiple types (e.g. MCQs and flashcards), call multiple tools
- In the Final Answer, present everything clearly with headers for each section

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)


# ── Agent Setup ───────────────────────────────────────────────────────────────

tools = [generate_mcq, generate_true_false, generate_flashcards]

react_agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=6,
)


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    my_notes = """
    Photosynthesis is the process by which green plants, algae, and some bacteria
    convert light energy (usually from the sun) into chemical energy stored as glucose.

    The overall equation is:
    6CO2 + 6H2O + light energy → C6H12O6 + 6O2

    It occurs in two stages:
    1. Light-dependent reactions (in the thylakoid membrane):
       - Absorb sunlight using chlorophyll
       - Split water molecules (photolysis), releasing oxygen as a byproduct
       - Produce ATP and NADPH

    2. Light-independent reactions / Calvin Cycle (in the stroma):
       - Use ATP and NADPH from stage 1
       - Fix CO2 into organic molecules
       - Produce glucose (G3P intermediate)

    Chlorophyll is the main pigment. It absorbs red and blue light best,
    and reflects green light (which is why plants appear green).

    Factors affecting the rate of photosynthesis:
    - Light intensity
    - CO2 concentration
    - Temperature
    - Water availability
    """

    result = agent_executor.invoke({
        "input": f"Generate MCQ questions and flashcards from my notes:\n{my_notes}"
    })

    print("\n" + "="*60)
    print("FINAL OUTPUT")
    print("="*60)
    print(result["output"])
