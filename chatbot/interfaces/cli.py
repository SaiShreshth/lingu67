"""
CLI Interface - Command-line interface for the chatbot.

Simple interactive CLI using the ChatOrchestrator.
"""

import sys
import os
import logging
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from chatbot.orchestrator.core import ChatOrchestrator, ChatSession

logger = logging.getLogger(__name__)


class CLI:
    """Command-line interface for the chatbot."""
    
    # ANSI colors
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    DIM = "\033[90m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    def __init__(self, orchestrator: Optional[ChatOrchestrator] = None):
        """
        Initialize CLI.
        
        Args:
            orchestrator: Optional ChatOrchestrator instance
        """
        self.orchestrator = orchestrator
        self.session = ChatSession(session_id="cli_session")
        
    def _init_orchestrator(self):
        """Lazy initialize orchestrator."""
        if self.orchestrator is None:
            print(f"{self.YELLOW}Initializing...{self.RESET}")
            self.orchestrator = ChatOrchestrator()
            print(f"{self.GREEN}Ready!{self.RESET}\n")
    
    def run(self):
        """Run the interactive CLI."""
        self._print_header()
        self._init_orchestrator()
        
        while True:
            try:
                # Get user input
                user_input = input(f"{self.BLUE}You:{self.RESET} ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit
                if user_input.lower() in ("exit", "quit", "bye"):
                    print(f"\n{self.GREEN}Goodbye!{self.RESET}")
                    break
                
                # Check for file upload command
                if user_input.startswith("/upload "):
                    self._handle_upload(user_input[8:].strip())
                    continue
                
                if user_input == "/files":
                    self._handle_list_files()
                    continue
                
                if user_input.startswith("/delete "):
                    self._handle_delete(user_input[8:].strip())
                    continue
                
                # Process through orchestrator
                print(f"{self.DIM}Thinking...{self.RESET}", end="\r")
                
                response = self.orchestrator.process(
                    query=user_input,
                    session=self.session,
                    stream=False
                )
                
                # Clear "Thinking..." and print response
                print(" " * 20, end="\r")
                print(f"{self.GREEN}AI:{self.RESET} {response.content}")
                
                # Show debug info if verbose
                if os.environ.get("VERBOSE"):
                    print(f"{self.DIM}[Intent: {response.intent.name}, Agents: {response.agents_used}]{self.RESET}")
                
            except KeyboardInterrupt:
                print(f"\n{self.YELLOW}Interrupted. Type 'exit' to quit.{self.RESET}")
            except Exception as e:
                print(f"{self.RED}Error: {e}{self.RESET}")
                logger.exception("CLI error")
    
    def _print_header(self):
        """Print welcome header."""
        print(f"""
{self.BOLD}╔══════════════════════════════════════════╗
║     Lingu67 Memory Assistant (v2.0)      ║
║        Orchestrator Architecture         ║
╚══════════════════════════════════════════╝{self.RESET}

{self.DIM}Commands:
  /upload <path>  - Upload a file
  /files          - List uploaded files
  /delete <name>  - Delete a file
  exit            - Quit{self.RESET}
""")
    
    def _handle_upload(self, path: str):
        """Handle file upload."""
        if not os.path.exists(path):
            print(f"{self.RED}File not found: {path}{self.RESET}")
            return
        
        print(f"{self.YELLOW}Uploading {os.path.basename(path)}...{self.RESET}")
        
        try:
            result = self.orchestrator.ingest_file(path)
            chunks = result.get("chunks", 0)
            print(f"{self.GREEN}✓ Uploaded ({chunks} chunks){self.RESET}")
        except Exception as e:
            print(f"{self.RED}Upload failed: {e}{self.RESET}")
    
    def _handle_list_files(self):
        """List uploaded files."""
        files = self.orchestrator.list_files()
        
        if not files:
            print(f"{self.DIM}No files uploaded yet.{self.RESET}")
            return
        
        print(f"\n{self.BOLD}Uploaded Files:{self.RESET}")
        for f in files:
            name = f.get("name", "unknown")
            chunks = f.get("chunks", 0)
            print(f"  • {name} ({chunks} chunks)")
        print()
    
    def _handle_delete(self, name: str):
        """Handle file deletion."""
        file_agent = self.orchestrator.agents.get("file")
        if file_agent:
            success = file_agent.delete_file(name)
            if success:
                print(f"{self.GREEN}✓ Deleted {name}{self.RESET}")
            else:
                print(f"{self.RED}Failed to delete {name}{self.RESET}")


def main():
    """Entry point for CLI."""
    logging.basicConfig(
        level=logging.INFO if os.environ.get("VERBOSE") else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    cli = CLI()
    cli.run()


if __name__ == "__main__":
    main()
