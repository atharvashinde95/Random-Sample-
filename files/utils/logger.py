"""
Rich-based terminal logger.
All output goes through here so the visual style is consistent.
"""
from __future__ import annotations
from typing import Any
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def banner() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]PRODUCTION SCHEDULING VALIDATION MVP[/bold cyan]",
            border_style="cyan",
        )
    )


def section(title: str) -> None:
    console.print(f"\n[bold white]--- {title} ---[/bold white]")


def info(label: str, message: str) -> None:
    console.print(f"[bold green][{label}][/bold green]  {message}")


def warn(label: str, message: str) -> None:
    console.print(f"[bold yellow][{label}][/bold yellow]  {message}")


def error(label: str, message: str) -> None:
    console.print(f"[bold red][{label}][/bold red]  {message}")


def saved(path: str) -> None:
    console.print(f"[bold blue][SAVED][/bold blue]   {path}")


def graph(message: str) -> None:
    console.print(f"[bold magenta][GRAPH][/bold magenta] {message}")


def react_thought(text: str) -> None:
    console.print(f"[bold yellow][CSV AGENT - THOUGHT][/bold yellow]    {text}")


def react_action(text: str) -> None:
    console.print(f"[bold cyan][CSV AGENT - ACTION][/bold cyan]     {text}")


def react_observation(text: str) -> None:
    console.print(f"[bold white][CSV AGENT - OBSERVATION][/bold white]\n  {text}")


def react_final(text: str) -> None:
    console.print(
        Panel(
            f"[bold green]{text}[/bold green]",
            title="[bold green]CSV AGENT - FINAL ANSWER[/bold green]",
            border_style="green",
        )
    )


def final_summary(summary: dict[str, Any]) -> None:
    section("FINAL SUMMARY")
    lines = [
        ("Problem Type", summary.get("problem_type", "?")),
        ("CP-SAT Compatible", "YES" if summary.get("is_cpsat_compatible") else "NO"),
        ("Objective", summary.get("objective", "?")),
        ("Constraints Valid", "YES" if summary.get("constraints_valid") else "NO"),
        ("Contradictions", "YES" if summary.get("contradictions_found") else "NO"),
        ("CSV Status", summary.get("csv_validation_status", "?")),
        ("Ready for CP-SAT", "YES" if summary.get("ready_for_cpsat") else "NO"),
    ]
    for label, value in lines:
        color = "green" if value in ("YES", "VALID") else ("red" if value in ("NO", "INVALID") else "white")
        console.print(f"[bold]{label:<22}[/bold][{color}]{value}[/{color}]")
