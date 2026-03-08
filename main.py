from __future__ import annotations

# -----------------------------------------------------------------------------
# Standard library imports
# -----------------------------------------------------------------------------
# Used for environment variable access and type hints
import os
from typing import List

# -----------------------------------------------------------------------------
# Third-party libraries
# -----------------------------------------------------------------------------
# dotenv allows us to load environment variables from a .env file
# This is standard practice when working with API keys
from dotenv import load_dotenv

# Pydantic provides strongly typed data models with validation
# This ensures the LLM output matches a predictable structure
from pydantic import BaseModel, Field, ValidationError

# -----------------------------------------------------------------------------
# LangChain imports
# -----------------------------------------------------------------------------
# init_chat_model is the modern LangChain entry point for initializing LLMs
# It provides a provider-agnostic interface across OpenAI, Anthropic, Ollama, etc.
from langchain.chat_models import init_chat_model

# ChatPromptTemplate provides structured prompts for chat models
from langchain_core.prompts import ChatPromptTemplate


# -----------------------------------------------------------------------------
# Environment validation helper
# -----------------------------------------------------------------------------
def require_env_var(variable_name: str) -> str:
    """
    Validate that a required environment variable exists.

    Why this matters:
    Production AI applications depend heavily on environment configuration
    (API keys, service URLs, feature flags). Failing early prevents obscure
    runtime failures later in the pipeline.
    """
    variable_value = os.getenv(variable_name, "").strip()

    if not variable_value:
        raise RuntimeError(
            f"Required environment variable '{variable_name}' is not set. "
            f"Add it to your .env file before running this script."
        )

    return variable_value


# -----------------------------------------------------------------------------
# Structured output schema definitions
# -----------------------------------------------------------------------------
# Instead of returning raw text from the LLM, we define structured data models.
# This allows the model response to be validated and used programmatically.
#
# This pattern is common in production LLM systems where downstream systems
# expect predictable data formats (APIs, dashboards, pipelines).


class TimelineEvent(BaseModel):
    """
    Represents one event in a person's career timeline.
    """
    year: str = Field(description="The year or approximate date of the event")
    event: str = Field(description="A concise description of what happened")
    significance: str = Field(description="Why the event matters")


class BiographyAnalysis(BaseModel):
    """
    Structured output schema representing a full biography analysis.

    Using Pydantic schemas with LLMs improves reliability because the
    generated output must conform to the expected structure.
    """
    person_name: str = Field(description="Full name of the person")

    executive_summary: str = Field(
        description="A concise 2-4 sentence summary"
    )

    key_organisations: List[str] = Field(
        default_factory=list,
        description="Major companies, organisations, or institutions mentioned"
    )

    timeline: List[TimelineEvent] = Field(
        default_factory=list,
        description="Chronological list of major events"
    )

    major_themes: List[str] = Field(
        default_factory=list,
        description="High-level themes such as entrepreneurship, politics, controversy"
    )

    controversial_points: List[str] = Field(
        default_factory=list,
        description="Important controversial or polarising topics mentioned in the text"
    )

    interview_questions: List[str] = Field(
        default_factory=list,
        description="Good follow-up questions someone could ask about this biography"
    )


# -----------------------------------------------------------------------------
# Input data
# -----------------------------------------------------------------------------
def get_biography_text() -> str:
    return """
    Harrison Chase is a software engineer and entrepreneur best known as the
    creator of LangChain, one of the most influential open-source frameworks
    for building applications powered by large language models (LLMs).

    Chase studied computer science and began his career as a machine learning
    engineer at Kensho Technologies, a financial AI company later acquired by
    S&P Global. At Kensho, he worked on systems that combined machine learning
    with structured financial data and real-world analytical workflows. This
    experience exposed him to a core challenge in applied AI: integrating
    machine learning models with existing software systems and external data.

    In late 2022, shortly after OpenAI released ChatGPT and generative AI began
    capturing global attention, Chase launched the LangChain project. His goal
    was to provide a developer framework that made it easier to connect large
    language models with external tools, APIs, documents, and databases.

    LangChain introduced a number of abstractions that quickly became standard
    concepts in the LLM developer ecosystem. These included prompts, chains,
    agents, tools, memory modules, and retrieval pipelines. These abstractions
    allowed developers to orchestrate complex AI workflows while keeping code
    modular and maintainable.

    As the project gained traction, LangChain became one of the fastest-growing
    open-source AI frameworks in the world. Within months, thousands of
    developers began using it to build chatbots, document analysis systems,
    research assistants, coding copilots, and enterprise automation tools.

    The rapid adoption of LangChain also sparked debates within the AI
    engineering community. Some critics argued that the framework introduced
    too much abstraction and could make debugging complex systems more
    difficult. Others praised it for accelerating experimentation and lowering
    the barrier to entry for developers working with large language models.

    In response to the growing complexity of AI agents, Chase and the LangChain
    team later introduced LangGraph, a framework designed for building
    stateful, deterministic agent workflows using graph-based execution.
    LangGraph aimed to improve reliability and control in AI systems by making
    reasoning flows explicit and traceable.

    The ecosystem expanded further with the creation of LangSmith, a platform
    designed for debugging, evaluating, and monitoring LLM applications in
    production. LangSmith allows developers to inspect agent reasoning traces,
    measure performance, and improve prompt reliability.

    Under Chase's leadership, LangChain evolved from a small open-source
    experiment into a full AI developer platform used by startups, research
    labs, and major technology companies. The framework now supports numerous
    model providers including OpenAI, Anthropic, Google, and local models
    running through tools such as Ollama.

    Chase has spoken publicly about the importance of reliability and
    observability in AI systems, arguing that building real-world AI products
    requires much more than simply calling an LLM API. His work has focused on
    helping developers design systems that combine reasoning, retrieval,
    external tools, and deterministic workflows.

    As the generative AI field continues to evolve, Chase remains a central
    figure in shaping how developers build applications on top of large
    language models. Many modern AI systems — including research assistants,
    autonomous agents, and knowledge retrieval tools — rely on patterns that
    were popularized by the LangChain ecosystem.
    """.strip()

# -----------------------------------------------------------------------------
# Prompt construction
# -----------------------------------------------------------------------------
def build_prompt() -> ChatPromptTemplate:
    """
    Creates the prompt template used to instruct the language model.

    Prompt design is critical in LLM applications. Here we:
    - constrain the model to use only provided information
    - prevent hallucinations
    - request structured analytical output
    """

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful biography analysis assistant.\n"
                "Use ONLY the information provided by the user.\n"
                "Do not add outside facts.\n"
                "If something is uncertain or not explicitly stated, omit it.\n"
                "Focus on factual extraction, chronology, and clear analytical structure."
            ),
            (
                "human",
                "Analyse the biography below.\n\n"
                "Return:\n"
                "1. The person's full name\n"
                "2. A concise executive summary\n"
                "3. Key organisations mentioned\n"
                "4. A timeline of major events\n"
                "5. Major themes in the person's career and public life\n"
                "6. Notable controversial or polarising points\n"
                "7. Five strong follow-up interview questions\n\n"
                "BIOGRAPHY:\n{biography_text}"
            ),
        ]
    )


# -----------------------------------------------------------------------------
# Model initialization
# -----------------------------------------------------------------------------
def build_model():
    """
    Initialize the LLM using LangChain's provider-agnostic interface.

    `init_chat_model` allows the same code to work with multiple LLM providers
    simply by changing the model name.
    """

    return init_chat_model(
        model="gpt-4.1",
        temperature=0,  # deterministic output for analysis tasks
    )


# -----------------------------------------------------------------------------
# Chain construction
# -----------------------------------------------------------------------------
def build_chain():
    """
    Constructs the LangChain pipeline.

    Pipeline architecture:

        Prompt Template
              │
              ▼
        Chat Model
              │
              ▼
    Structured Output Parser (Pydantic)

    The final result is a validated Python object.
    """

    prompt = build_prompt()

    # Bind the structured schema to the model
    model = build_model().with_structured_output(BiographyAnalysis)

    # LCEL composition operator builds the pipeline
    return prompt | model


# -----------------------------------------------------------------------------
# Output formatting
# -----------------------------------------------------------------------------
def print_analysis(analysis: BiographyAnalysis) -> None:
    """
    Nicely formats the analysis results for terminal output.

    Separating presentation from logic keeps the architecture clean
    and makes it easier to replace the output layer later with:
    - a web interface
    - a REST API
    - a data pipeline
    """

    print("\n" + "=" * 80)
    print(f"BIOGRAPHY ANALYSIS: {analysis.person_name}")
    print("=" * 80)

    print("\nEXECUTIVE SUMMARY")
    print("-" * 80)
    print(analysis.executive_summary)

    print("\nKEY ORGANISATIONS")
    print("-" * 80)
    for organisation_name in analysis.key_organisations:
        print(f"- {organisation_name}")

    print("\nTIMELINE")
    print("-" * 80)
    for timeline_event in analysis.timeline:
        print(f"- {timeline_event.year}: {timeline_event.event}")
        print(f"  Why it matters: {timeline_event.significance}")

    print("\nMAJOR THEMES")
    print("-" * 80)
    for theme in analysis.major_themes:
        print(f"- {theme}")

    print("\nCONTROVERSIAL / POLARISING POINTS")
    print("-" * 80)
    for point in analysis.controversial_points:
        print(f"- {point}")

    print("\nFOLLOW-UP INTERVIEW QUESTIONS")
    print("-" * 80)
    for index, question in enumerate(analysis.interview_questions, start=1):
        print(f"{index}. {question}")


# -----------------------------------------------------------------------------
# Main execution entry point
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Orchestrates the entire analysis workflow.

    Steps:
    1. Load environment configuration
    2. Validate API credentials
    3. Build the LLM pipeline
    4. Run inference
    5. Display results
    """

    load_dotenv()

    # Ensure required API key exists
    require_env_var("OPENAI_API_KEY")

    biography_text = get_biography_text()

    # Build the LangChain pipeline
    chain = build_chain()

    try:
        # Execute the chain with input data
        analysis = chain.invoke({"biography_text": biography_text})

    except ValidationError as validation_error:
        raise RuntimeError(
            "The model response did not match the expected structured schema."
        ) from validation_error

    except Exception as exc:
        raise RuntimeError(
            "Biography analysis failed. Check your model access, environment "
            "variables, and installed package versions."
        ) from exc

    # Display formatted output
    print_analysis(analysis)


# Standard Python entry point
if __name__ == "__main__":
    main()