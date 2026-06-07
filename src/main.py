"""
main.py — narrative-playlist-agent
Entry point CLI. Interfaz de usuario con Rich.

Uso:
    python src/main.py
    python src/main.py --prompt "Playlist oscura que suba a épica"
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

# Añadir src al path para imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.agent import NarrativePlaylistAgent

console = Console()


# ─────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────

def print_header():
    console.print(Panel.fit(
        "[bold cyan]narrative-playlist-agent[/bold cyan]\n"
        "[dim]Arcos emocionales → playlists matemáticamente validadas[/dim]",
        border_style="cyan"
    ))


def print_result(result: dict):
    """Imprime el resultado completo del agente con Rich."""

    # Respuesta narrativa del agente
    if result.get("final_response"):
        console.print(
            Panel(
                result["final_response"],
                title="[bold green]🎵 Playlist generada[/bold green]",
                border_style="green",
                padding=(1, 2)
            )
        )

    # Info de archivos generados
    csv_path = result.get("csv_path")
    json_path = result.get("json_path")

    if csv_path or json_path:
        console.print("\n[bold]📄 Archivos generados:[/bold]")
        if csv_path:
            console.print(f"  CSV  → [cyan]{csv_path}[/cyan]")
        if json_path:
            console.print(f"  JSON → [cyan]{json_path}[/cyan]")

        console.print("\n[bold]🎵 Para escucharla:[/bold]")
        console.print("  1. Ve a [cyan]tunemymusic.com[/cyan]")
        console.print("  2. Selecciona [bold]File[/bold] como fuente")
        console.print("  3. Sube el archivo [bold].csv[/bold]")
        console.print("  4. Elige tu plataforma: Spotify, Apple Music, YouTube Music...")
        console.print("  5. La playlist aparecerá en tu cuenta\n")

    # Estadísticas de ejecución
    iterations = result.get("iterations_used", 0)
    if iterations:
        console.print(
            f"[dim]Iteraciones usadas: {iterations}[/dim]"
        )

    # Warning si hubo error
    if result.get("error") == "max_iterations_reached":
        console.print(
            "\n[yellow]⚠ El agente alcanzó el límite de iteraciones. "
            "Intenta con un prompt más simple.[/yellow]"
        )


def get_user_prompt(preset: str = None) -> str:
    """Obtiene el prompt del usuario — desde argumento o input interactivo."""
    if preset:
        return preset

    console.print("\n[bold]Describe el arco emocional de tu playlist:[/bold]")
    console.print(
        "[dim]Ejemplos:\n"
        "  • 'Playlist de 9 canciones: oscura → épica → paz'\n"
        "  • 'Playlist para estudiar, concentrada y tranquila, 6 tracks'\n"
        "  • 'Viaje emocional de tristeza profunda a euforia pura'\n"
        "[/dim]"
    )

    prompt = console.input("[bold cyan]→ [/bold cyan]").strip()
    if not prompt:
        console.print("[red]El prompt no puede estar vacío.[/red]")
        sys.exit(1)

    return prompt


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="narrative-playlist-agent — Arcos emocionales en playlists"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help="Prompt del arco emocional (si se omite, se pide interactivo)"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path a la DB SQLite (opcional, usa la default si se omite)"
    )
    args = parser.parse_args()

    print_header()

    try:
        # Inicializar agente
        agent_kwargs = {}
        if args.db:
            agent_kwargs["db_path"] = args.db

        agent = NarrativePlaylistAgent(**agent_kwargs)

        # Obtener prompt
        user_prompt = get_user_prompt(args.prompt)

        console.print(
            f"\n[dim]Prompt: {user_prompt}[/dim]\n"
        )

        # Ejecutar agente
        result = agent.run(user_prompt)

        # Mostrar resultado
        print_result(result)

    except FileNotFoundError as e:
        console.print(f"\n[red]❌ {e}[/red]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"\n[red]❌ Configuración: {e}[/red]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrumpido.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]❌ Error inesperado: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
