# Task image + the agent CLIs. Built by `fsbench build`; never contains grader/ or reference/.
ARG TASK_IMAGE
FROM node:20-bullseye-slim AS node

FROM ${TASK_IMAGE}
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

ARG CLAUDE_CODE_VERSION=latest
ARG CODEX_VERSION=latest
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION} @openai/codex@${CODEX_VERSION} \
 && npm cache clean --force

# The harness runs the container as the invoking (non-root) user with HOME here.
RUN mkdir -p /tmp/agent-home && chmod 1777 /tmp/agent-home
