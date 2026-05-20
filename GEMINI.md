# Project Instructions: QUBE RL

## Communication & Workflow
- **Verbose Reasoning:** Always explain **what** you are planning to change and, more importantly, **why**. Provide technical rationale for every architectural or logic adjustment.
- **Empirical Rigor:** Never presume model behavior or state conclusions about performance without specific quantitative data (e.g., reward means, standard deviations, step counts). If the data is unavailable or insufficient, explicitly state your conclusion as an **assumption** or **hypothesis**. 
- **Mandatory Confirmation:** Do NOT modify any code files until the user has explicitly confirmed the proposed plan. 
- **Peer Programming Tone:** Maintain a collaborative, senior-engineer-to-peer tone.

## Context Management
- **Performance Awareness:** Be mindful of context bloat. Use surgical file reads (`start_line`/`end_line`) and avoid reading large data files (.csv, .zip).
- **Persistence:** Use `MEMORY.md` in the private memory folder to store long-term project state and "threads" so they survive session restarts.
- **Final Model Target:** The folder `FromYousseff/Lasthopemodel/` is the official repository for production-ready, validated models. Always look here for the current deployment candidate.
