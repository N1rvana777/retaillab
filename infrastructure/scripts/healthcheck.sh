#!/bin/bash
# Docker HEALTHCHECK script
curl -sf http://localhost:8000/api/v1/health || exit 1
