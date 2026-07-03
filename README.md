Full project on Notion: https://caramel-room-ed9.notion.site/Sales-Automated-Pipeline-38f86c98b25280dabbdac5fb6f8400a0?source=copy_link

## Overview

This project is an AI-powered SDR recommendation agent that retrieves prospect and activity data from a Notion database, analyzes sales context using LLMs, and generates personalized outreach recommendations.

The project demonstrates API integration, workflow automation, and LLM orchestration for sales operations.

## Notion Database

This project uses the Notion API as its primary data source.

The agent reads prospect information from a Notion database, processes each record, and writes AI-generated recommendations back into Notion.

## Setup

Create a `.env` file containing:

```env
NOTION_API_KEY=your_notion_api_key
NOTION_DATABASE_ID=your_database_id
OPENAI_API_KEY=your_openai_api_key

## Features

- Retrieves prospect data from Notion via the Notion API
- Processes records with Python
- Generates AI-powered SDR recommendations
- Writes recommendations back to Notion
- Includes automated workflow for recurring recommendation generation


