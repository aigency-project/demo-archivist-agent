# Reception Agent - Front Desk Assistant for the Detective Agency

This example demonstrates a polite, lightweight receptionist agent built with the Aigency framework. The Reception Agent greets users, asks brief clarifying questions, and routes them conceptually to the appropriate specialist (no MCP services required).

## 🕵️ System Architecture

### Specialized Agents

1. **`reception_agent`** - Front Desk Assistant
   - Greets visitors warmly and keeps interactions efficient
   - Always responds in the user's language
   - Asks concise clarifying questions to understand the need
   - Explains which specialist is best suited for the case
   - Provides clear next steps

### MCP Services (Model Context Protocol)

- None. This agent intentionally runs without MCP tools to stay simple and fast.

## 🚀 How to Run

### Prerequisites

1. Docker and Docker Compose installed
2. Environment variables configured in `.env`:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key
   GOOGLE_GENAI_USE_VERTEXAI=FALSE
   ```

### Run

```bash
# From the reception_agent directory
docker-compose up --build
```

### Access Ports

- **Reception Agent**: http://localhost:8082 (Main entry point)
- **A2A Inspector**: http://localhost:6007

## 💼 Use Cases

### 1. Greeting and Quick Triage
Briefly understand the visitor's need and propose the right specialist.

**Example Prompt:**
```
"Hello! I need help with a case involving stolen documents."
```
**Agent's Likely Response:**
"Welcome! I'd be happy to help. To direct you correctly, could you share one detail: do you already have any evidence (e.g., documents, fingerprints, video)? If so, the Case Agent is your best fit. Otherwise, if this involves multiple stakeholders, the Detective Manager can coordinate a broader investigation. Next steps: 1) Describe key evidence, 2) Share timeline, 3) Identify known witnesses."

### 2. Clarifying Question Before Routing
Ask one concise question if the request is ambiguous.

**Example Prompt:**
```
"We suspect someone inside the company leaked information."
```
**Agent's Likely Response:**
"Understood. One quick question: do you need contacts or credibility checks from an informant network? If yes, the Informant Agent is ideal. If you need evidence analysis and suspect profiling, the Case Agent will take the lead."

## 🔧 Agent Configuration

### Reception Agent
- **Specialization**: Greeting, triage, routing, and providing next steps
- **MCP Tools**: None
- **Skills**: `Greet and Route`, `Provide Next Steps`

## 📊 Monitoring and Observability

- **A2A Inspector**: Agent inspection tools at http://localhost:6007
- **Logs**: The agent generates logs for tracking and debugging

## 🔍 Interaction Examples

### Complete Example
```
User: "Hi, I have a complex fraud case across multiple departments."

Reception Agent:
1. Greets and summarizes: "Welcome! You mentioned a complex fraud case affecting multiple departments."
2. Triage: "For coordinated, multi-threaded investigations, the Detective Manager is the best fit."
3. Next steps: "Please provide a brief timeline, known involved teams, and any available documents."
```

### Quick Evidence Routing
```
User: "I found fingerprints and a security video after a break-in."

Reception Agent:
1. Acknowledge: "Thank you—those are key evidences."
2. Route: "The Case Agent will analyze physical and digital evidence and build a suspect profile."
3. Next steps: "Please share the files and any witness statements."
```

## 🛠️ Extensibility

The system can be easily expanded:

- **Add MCPs later**: Evidence database lookup, appointment scheduling, intake forms
- **New Skills**: Intake checklist builder, case summary generator
- **New Integrations**: Connect to the detective agency’s case management system

## 📝 Development Notes

- The agent always responds in the user's language (e.g., Spanish, English)
- It prioritizes clarity and brevity, and does not perform deep investigations
- Designed to be simple and fast to deploy (no MCPs)

## 🔐 Content & Safety Considerations

- Never request or reveal real personal information beyond what’s necessary for triage
- Maintain a respectful, helpful, and professional tone
- Keep guidance high-level and avoid sensitive details in public logs
