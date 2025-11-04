import { useEffect } from 'react';

import { usePathname } from 'next/navigation';

import { useToast } from '@/hooks/use-toast';
import { useAgentStore } from '@/lib/store/agent/agentStore';

export const useFetchMessages = () => {
  const { toast } = useToast();
  const path = usePathname();
  const chatId = path.split('/')[2];
  const { fetchMessages, messages, conversationsLoaded } = useAgentStore();

  useEffect(() => {
    // Cleanup flag
    let isMounted = true;

    if (!(chatId && conversationsLoaded)) return;

    const loadMessages = async () => {
      try {
        await fetchMessages(chatId);
      } catch (err: any) {
        if (!isMounted) return;
        toast({
          variant: 'destructive',
          title: 'Error loading messages',
          description: err.message || 'Could not retrieve chat messages.',
        });
      }
    };
    loadMessages();

    // cleanup function
    return () => {
      isMounted = false;
    };
  }, [chatId, conversationsLoaded, fetchMessages, toast]);

  return messages;
};
