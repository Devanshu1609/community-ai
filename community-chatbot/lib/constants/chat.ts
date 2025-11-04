import type { IntegrationMode } from '@/types/chat/types';

export const integrationModes: IntegrationMode[] = [
	{
		id: "general",
		name: "General Assistant",
		icon: "🤖",
		image: "/mifos.png",
		color: "bg-blue-600 hover:bg-blue-700 text-white",
		description: "Ask about features, documentation, or community",
		systemPrompt:
			"You are a helpful Mifos Community assistant. Help with general questions about Mifos features, documentation, and community support.",
	},
	{
		id: "slack",
		name: "Slack Assistant",
		icon: "💬",
		image: "/slack.png",
		color: "bg-purple-600 hover:bg-purple-700 text-white",
		description: "Get channel info, user details, or search messages",
		systemPrompt:
			"You are a Slack integration assistant. Help users with Slack channel management, user queries, message searches, and workspace administration.",
	},
	{
		id: "jira",
		name: "Jira Assistant",
		icon: "🔷",
		image: "/jira.svg",
		color: "bg-blue-500 hover:bg-blue-600 text-white",
		description: "Check ticket status, create issues, or view reports",
		systemPrompt:
			"You are a Jira integration assistant. Help users with ticket management, issue creation, project tracking, and generating reports.",
	},
	{
		id: "github",
		name: "GitHub Assistant",
		icon: "🐙",
		image: "/github.png",
		color: "bg-purple-600 hover:bg-purple-700 text-white",
		description: "Review PRs, check issues, or get repo info",
		systemPrompt:
			"You are a GitHub integration assistant. Help users with pull request reviews, issue tracking, repository management, and code collaboration.",
	},
];

export const quickActions = [
	"How to create new client?",
	"API documentation",
	"Troubleshoot reporting",
	"Mobile banking setup",
];
