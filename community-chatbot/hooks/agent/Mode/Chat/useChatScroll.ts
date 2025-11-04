import { useEffect, useRef } from 'react';

import { useAgentStore } from '@/lib/store/agent/agentStore';

export const useChatScroll = () => {
  const scrollAreaRef = useRef<HTMLDivElement | null>(null);
  const { messages, status } = useAgentStore();

  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollElement = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]') as HTMLElement
      if (scrollElement) {
        scrollElement.scrollTo({
          top: scrollElement.scrollHeight,
          behavior: 'smooth'
        })
      }
    }
  }, [messages, status]);

  return scrollAreaRef;
};
