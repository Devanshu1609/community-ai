export interface Message {
    id: string;
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: number;
}

export interface ChatHistoryItem {
    id: string;
    title: string;
    createdAt: Date;
    updatedAt: Date;
    icon: string;
    messages: Message[];
    mode: string;
    userId: string;
}

export interface IntegrationMode {
    id: string;
    name: string;
    icon: string;
    image: string;
    color: string;
    description: string;
    systemPrompt?: string;
    useCustomBackend?: boolean;
}

export interface Conversation extends Omit<ChatHistoryItem, 'messages'> {
    messages: Message[];
}

export type ChatStatus = 'ready' | 'submitted' | 'streaming' | 'error';
