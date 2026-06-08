"""
agent/agent.py — narrative-playlist-agent
Loop principal del agente. Orquesta LLM ↔ tools ↔ validación.

Responsabilidades:
  - Construir y mantener el contexto de mensajes
  - Llamar al LLM (Groq via OpenAI SDK)
  - Detectar tool calls y despacharlos al módulo correcto
  - Correr playlist_validator sobre el output
  - Retornar PlaylistOutput validado

Principio de harness:
  El LLM razona. Python controla.
  Todo lo que puede fallar de forma determinista está fuera del LLM.
"""

import json
import os
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from src.agents.system_prompt import SYSTEM_PROMPT
from src.agents.tools import (
    query_song_database,
    check_db_ready,
    TOOL_SCHEMA,
)
from src.agents.output_tools import (
    generate_playlist_file,
    GENERATE_PLAYLIST_SCHEMA,
)
from src.agents.db_tools import (
    add_track_to_database,
    check_track_exists,
    ADD_TRACK_SCHEMA,
    CHECK_TRACK_SCHEMA,
)

load_dotenv()

console = Console()

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "db" / "tracks.db"

LLM_MODEL = "llama-3.3-70b-versatile"
MAX_ITERATIONS = 15      # guard contra loops infinitos
MAX_TOKENS = 2000

# Todas las tools disponibles para el LLM
ALL_TOOLS = [
    TOOL_SCHEMA,
    ADD_TRACK_SCHEMA,
    CHECK_TRACK_SCHEMA,
    GENERATE_PLAYLIST_SCHEMA,
]

# Mapa de tool_name → función Python
TOOL_DISPATCH = {
    "query_song_database":    query_song_database,
    "add_track_to_database":  add_track_to_database,
    "check_track_exists":     check_track_exists,
    "generate_playlist_file": generate_playlist_file,
}


# ─────────────────────────────────────────────
# AGENTE
# ─────────────────────────────────────────────

class NarrativePlaylistAgent:
    """
    Agente de curación musical con razonamiento secuencial.

    Uso:
        agent = NarrativePlaylistAgent()
        result = agent.run("Playlist oscura que suba a épica y termine en paz")
        print(result["arc_summary"])
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.client = self._build_llm_client()
        self._verify_db()

    def _build_llm_client(self) -> OpenAI:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY no encontrada. "
                "Asegúrate de tener un archivo .env con GROQ_API_KEY=..."
            )
        return OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )

    def _verify_db(self) -> None:
        ok, msg = check_db_ready(self.db_path)
        if not ok:
            raise FileNotFoundError(
                f"Base de datos no disponible: {msg}\n"
                f"Ejecuta: python src/ingestion/load_dataset.py"
            )
        console.print(f"[dim]✓ {msg}[/dim]")

    # ─────────────────────────────────────────
    # LOOP PRINCIPAL
    # ─────────────────────────────────────────

    def run(self, user_prompt: str) -> dict:
        """
        Procesa el prompt del usuario y retorna el resultado completo.

        Args:
            user_prompt: descripción del arco emocional en lenguaje natural

        Returns:
            dict con playlist_output, csv_path, json_path, final_response
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt}
        ]

        iterations = 0
        playlist_output = None
        csv_path = None
        json_path = None
        forced_generate = False  # evita reintentar el guard en bucle infinito

        console.print(f"\n[bold cyan]🎵 Generando playlist...[/bold cyan]")

        while iterations < MAX_ITERATIONS:
            iterations += 1

            with Live(
                Spinner("dots", text=Text(
                    f"[dim]Turno {iterations} — pensando...[/dim]"
                )),
                console=console,
                transient=True
            ):
                response = self._call_llm(messages)

            message = response.choices[0].message

            # Sin tool calls → el LLM terminó, emite respuesta final
            if not message.tool_calls:
                # Guard: el modelo no puede declarar la playlist como lista
                # si nunca llamó a generate_playlist_file (csv_path sigue None).
                # Algunos modelos copian la plantilla de respuesta final como
                # texto sin ejecutar el PASO 3. Lo forzamos a generarla.
                if csv_path is None and not forced_generate:
                    forced_generate = True
                    messages.append(message)
                    messages.append({
                        "role": "user",
                        "content": (
                            "No has generado el archivo todavía. NO describas "
                            "la playlist como lista: debes llamar a la "
                            "herramienta generate_playlist_file con todas las "
                            "fases y tracks ANTES de dar la respuesta final."
                        )
                    })
                    continue

                final_text = message.content or ""
                console.print(f"\n[green]✓ Playlist completa[/green]")
                return {
                    "final_response": final_text,
                    "playlist_output": playlist_output,
                    "csv_path": csv_path,
                    "json_path": json_path,
                    "iterations_used": iterations
                }

            # Hay tool calls → ejecutar y continuar el loop
            messages.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_result = self._execute_tool(tool_call)

                # Capturar paths si generate_playlist_file fue exitoso
                if tool_name == "generate_playlist_file":
                    try:
                        result_data = json.loads(tool_result)
                        if result_data.get("success"):
                            csv_path = result_data.get("csv_path")
                            json_path = result_data.get("json_path")
                            playlist_output = result_data
                    except json.JSONDecodeError:
                        pass

                console.print(
                    f"  [dim]→ {tool_name} "
                    f"({'✓' if self._is_success(tool_result) else '✗'})[/dim]"
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })

        # Guard: máximo de iteraciones alcanzado
        console.print(
            f"[yellow]⚠ Máximo de iteraciones ({MAX_ITERATIONS}) alcanzado[/yellow]"
        )
        return {
            "final_response": (
                "El agente alcanzó el límite de iteraciones. "
                "Intenta con un arco más simple o menos canciones."
            ),
            "playlist_output": playlist_output,
            "csv_path": csv_path,
            "json_path": json_path,
            "iterations_used": iterations,
            "error": "max_iterations_reached"
        }

    # ─────────────────────────────────────────
    # LLAMADA AL LLM
    # ─────────────────────────────────────────

    def _call_llm(self, messages: list[dict]):
        """Llama al LLM con retry básico ante errores de rate limit."""
        import time

        for attempt in range(3):
            try:
                return self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    tools=ALL_TOOLS,
                    tool_choice="auto",
                    max_tokens=MAX_TOKENS,
                    temperature=0.7
                )
            except Exception as e:
                error_str = str(e).lower()
                if "rate_limit" in error_str and attempt < 2:
                    wait = (attempt + 1) * 5
                    console.print(
                        f"  [yellow]Rate limit — esperando {wait}s...[/yellow]"
                    )
                    time.sleep(wait)
                else:
                    raise

    # ─────────────────────────────────────────
    # DESPACHO DE TOOLS
    # ─────────────────────────────────────────

    def _execute_tool(self, tool_call) -> str:
        """
        Despacha un tool_call al módulo Python correcto.
        Siempre retorna JSON string — nunca lanza excepción al loop.
        """
        tool_name = tool_call.function.name

        try:
            raw_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            return json.dumps({
                "success": False,
                "error": f"Argumentos JSON inválidos para {tool_name}: {e}"
            })

        fn = TOOL_DISPATCH.get(tool_name)
        if fn is None:
            return json.dumps({
                "success": False,
                "error": f"Tool desconocida: {tool_name}. "
                         f"Tools disponibles: {list(TOOL_DISPATCH.keys())}"
            })

        # query_song_database y check_track_exists necesitan db_path
        if tool_name in ("query_song_database", "check_track_exists",
                         "add_track_to_database"):
            return fn(raw_args, db_path=self.db_path)

        return fn(raw_args)

    # ─────────────────────────────────────────
    # UTILIDADES
    # ─────────────────────────────────────────

    @staticmethod
    def _is_success(tool_result: str) -> bool:
        """Verifica rápidamente si el resultado de una tool fue exitoso."""
        try:
            data = json.loads(tool_result)
            return data.get("success", True) is not False
        except Exception:
            return True
