#!/bin/bash

# LLM-Powered Policy Q&A Assistant — Cleanup Script
# Removes all deployed Databricks resources for this accelerator

set -e

echo "🧹 Starting cleanup of Policy Q&A Assistant resources..."

# Check Databricks CLI is installed
if ! command -v databricks &> /dev/null; then
    echo "❌ Error: databricks CLI is not installed"
    echo "   Install with: pip install databricks-cli"
    exit 1
fi

# Validate bundle
echo "📋 Validating bundle configuration..."
databricks bundle validate

# Destroy all deployed resources
echo "🗑️  Destroying deployed jobs and notebooks..."
databricks bundle destroy --auto-approve

echo "✅ Cleanup completed successfully!"
echo ""
echo "📝 Summary:"
echo "  - Deployed jobs and notebooks removed"
echo "  - Unity Catalog tables are NOT automatically deleted"
echo ""
echo "  To remove Unity Catalog data manually, run in a Databricks notebook:"
echo "    DROP TABLE IF EXISTS <catalog>.policy_assistant.qa_results;"
echo "    DROP TABLE IF EXISTS <catalog>.policy_assistant.policy_embeddings;"
echo "    DROP TABLE IF EXISTS <catalog>.policy_assistant.policy_chunks;"
echo "    DROP TABLE IF EXISTS <catalog>.policy_assistant.policy_documents;"
echo "    DROP SCHEMA IF EXISTS <catalog>.policy_assistant CASCADE;"
echo "    DROP CATALOG IF EXISTS <catalog> CASCADE;"
