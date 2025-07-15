"""
Development Configuration - Enhanced Developer Experience (2025)

This module provides configuration for cutting-edge development tools:
- Hot reload for templates and static files
- Live code patching with Jurigged  
- Rich terminal output
- AI agent development setup
"""

import os
from typing import Dict, List, Optional

# Hot Reload Configuration
HOT_RELOAD_CONFIG = {
    "enabled": os.getenv("DEBUG", "false").lower() == "true",
    "watch_paths": [
        "src/big_mood_detector",
        "templates",
        "static",
        "tests",
    ],
    "watch_extensions": [".py", ".html", ".css", ".js", ".json", ".yaml", ".toml"],
    "ignore_paths": [
        "__pycache__",
        ".git",
        ".pytest_cache",
        "node_modules",
        ".mypy_cache",
    ],
}

# Arel Browser Hot Reload Script (for HTML templates)
AREL_HOT_RELOAD_SCRIPT = """
<script>
(() => {
    if (!window.location.hostname.includes('localhost') && 
        !window.location.hostname.includes('127.0.0.1')) {
        return; // Only run in development
    }
    
    const socketUrl = `ws://${window.location.host}/hot-reload`;
    let ws = new WebSocket(socketUrl);
    
    const reconnectInterval = 100;
    const maxReconnectAttempts = 10;
    let reconnectAttempts = 0;
    
    ws.addEventListener('close', () => {
        const reloadIfCanConnect = () => {
            reconnectAttempts++;
            if (reconnectAttempts > maxReconnectAttempts) {
                console.log('[Hot Reload] Max reconnection attempts reached');
                return;
            }
            
            const newWs = new WebSocket(socketUrl);
            newWs.addEventListener('error', () => {
                setTimeout(reloadIfCanConnect, reconnectInterval);
            });
            newWs.addEventListener('open', () => {
                console.log('[Hot Reload] Reconnected - reloading page');
                window.location.reload();
            });
        };
        
        console.log('[Hot Reload] Connection lost, attempting to reconnect...');
        setTimeout(reloadIfCanConnect, reconnectInterval);
    });
    
    ws.addEventListener('open', () => {
        console.log('[Hot Reload] Connected to development server');
    });
})();
</script>
"""

# AI Agent Development Configuration
AGENT_CONFIG = {
    "frameworks": {
        "langgraph": {
            "enabled": True,
            "description": "Graph-based agent workflows with OpenAI compatibility",
            "use_case": "Complex mood analysis pipelines with stateful flows",
        },
        "crewai": {
            "enabled": True, 
            "description": "Simple role-based agent teams",
            "use_case": "Collaborative mood assessment (data analyst + clinician + reviewer)",
        },
        "pydantic_ai": {
            "enabled": True,
            "description": "Type-safe, minimal agents by Pydantic team",
            "use_case": "Structured mood prediction outputs with validation",
        },
        "agno": {
            "enabled": True,
            "description": "Production-ready agentic memory", 
            "use_case": "Patient session memory and longitudinal tracking",
        },
    },
    "models": {
        "openai": ["gpt-4o", "gpt-4o-mini"],
        "anthropic": ["claude-3-5-sonnet-20241022"],
        "google": ["gemini-1.5-pro", "gemini-1.5-flash"],
        "local": ["ollama/llama3.1:8b"],
    },
    "vector_stores": {
        "chromadb": "Agent memory and clinical knowledge base",
        "faiss": "Fast similarity search for patient data",
    },
}

# Rich Terminal Configuration
RICH_CONFIG = {
    "theme": "monokai",
    "show_time": True,
    "show_path": True,
    "enable_color": True,
    "console_width": None,  # Auto-detect
}

# Development Environment Detection
def is_development() -> bool:
    """Check if we're in development mode."""
    return os.getenv("DEBUG", "false").lower() == "true"

def get_hot_reload_script() -> str:
    """Get the hot reload script for HTML templates."""
    if is_development():
        return AREL_HOT_RELOAD_SCRIPT
    return ""

def get_agent_frameworks() -> List[str]:
    """Get list of enabled AI agent frameworks."""
    enabled = []
    for name, config in AGENT_CONFIG["frameworks"].items():
        if config["enabled"]:
            enabled.append(name)
    return enabled

def print_dev_info():
    """Print development environment information."""
    if is_development():
        print("🔥 Development mode enabled!")
        print(f"📁 Watching paths: {', '.join(HOT_RELOAD_CONFIG['watch_paths'])}")
        print(f"🤖 Agent frameworks: {', '.join(get_agent_frameworks())}")
        print("⚡ Hot reload: Browser + live code patching active")
    else:
        print("🚀 Production mode") 