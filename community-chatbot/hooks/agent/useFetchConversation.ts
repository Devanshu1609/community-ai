'use client';

import { useEffect, useState } from 'react';

import { usePathname } from 'next/navigation';

import { useToast } from '@/hooks/use-toast';
import { useAgentStore } from '@/lib/store/agent/agentStore';

export function useFetchConversation() {
    const { toast } = useToast();
    const pathname = usePathname();
    const currentMode = pathname.split('/')[1];
    const { fetchConversations, currentUser, chats } = useAgentStore();
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        let isMounted = true;

        if (!currentUser) {
            setLoading(false);
            return;
        }

        const loadConversations = async () => {
            setLoading(true);
            try {
                await fetchConversations(currentUser, currentMode);
            } catch (err: any) {
                if (!isMounted) return;

                toast({
                    variant: 'destructive',
                    title: 'Error loading chats',
                    description: err.message || 'Could not retrieve your conversations.',
                });
            } finally {
                if (isMounted) {
                    setLoading(false);
                }
            }
        };
        loadConversations();

        return () => {
            isMounted = false;
        };
    }, [currentMode, currentUser, fetchConversations, toast]);

    return { chats, loading };
}
