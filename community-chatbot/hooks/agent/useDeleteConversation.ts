'use client'

import { usePathname, useRouter } from 'next/navigation';

import { useToast } from '@/hooks/use-toast';
import { useAgentStore } from '@/lib/store/agent/agentStore';

export function useDeleteConversation() {
    const router = useRouter();
    const { toast } = useToast();
    const pathname = usePathname();
    const currentMode = pathname.split('/')[1];
    const activeChatId = pathname.split('/')[2];
    const { deleteConversation } = useAgentStore();

    const remove = async (chatId: string) => {
        try {
            await deleteConversation(chatId);
            toast({ title: 'Chat deleted' });
            if (activeChatId === chatId) router.push(`/${currentMode}`);
        } catch (err: any) {
            toast({
                variant: 'destructive',
                title: 'Error deleting chat',
                description: err.message || 'Could not delete chat.',
            });
        }
    };

    return { remove };
}
