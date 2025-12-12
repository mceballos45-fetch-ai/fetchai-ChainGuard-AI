# ChainGuard-AI

## Slides
[Presentation slides](https://docs.google.com/presentation/d/1JvOaihwfq2KQz0abSDV2B8ZFJkr2lrd7ZzUGdlRPug8/edit)

# Getting Started

# Prerequisites
 
- **Supabase CLI:** You'll need the Supabase CLI installed. If you don't have it yet, follow the [installation guide](https://supabase.com/docs/guides/cli).
- **Python:** Make sure you have Python 3.7+ installed.

## Setup

### 1. Python Environment
Create a virtual environment and activate it:

```
python -m venv venv
Mac: source venv/bin/activate  
Windows: venv\Scripts\activate

```
Install dependencies:
```
pip install -r requirements.txt
```

## Starting supabase

```
bash

supabase start

```
##
 Running Agents

You can start the agents using Python module syntax:
```
bash

# Start the orchestrator agent

python -m agentverse.supplier_orchestrator

# Start the find-supplier agents

python -m agents.supplier_search.compliance_agent
python -m agents.supplier_search.financial_agent
python -m agents.find_supplier_agent

# Start the monitoring agents
python -m agents.monitoring.demand_agent
python -m agents.monitoring.logistics_agent

```

## Interacting with Agents
Once your agents are running, you can interact with the supplier_orchestrator agent via [asi:one](https://asi1.ai/chat)

## Docker
start supabase
```
supabase start
```

Build the image
```
docker build -f Dockerfile -t chainguard-orchestrator .
```
Run it

```
docker run --env-file .env -p 8080:8080 chainguard-agent
```
Note this multi-agent arch is to only containerise the orchestrator_agent. Users of ChainGuard-AI are expected to containerize and deploy the other agents themselves using YAML, which is why no Docker setup is provided for them.
