'use client'

import { useToast } from '@/hooks/use-toast';
import { useAgentStore } from '@/lib/store/agent/agentStore';

export function useRenameConversation() {
    const { renameConversation, chats } = useAgentStore();
    const { toast } = useToast();

    const rename = async (chatId: string, newTitle: string) => {
        const trimmed = newTitle.trim();
        const originalTitle = chats.find(c => c.id === chatId)?.title;

        if (!trimmed || trimmed === originalTitle) return;

        try {
            await renameConversation(chatId, trimmed);
            toast({ title: 'Chat renamed' });
        } catch (err: any) {
            toast({
                variant: 'destructive',
                title: 'Error renaming chat',
                description: err.message || 'Could not rename chat.',
            });
        }
    };

    return { rename };
}