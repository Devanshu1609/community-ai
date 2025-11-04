import {
  LoadingIndicator,
} from '@/components/agent/Mode/Chat/Panel/LoadingIndicator';
import { MessageList } from '@/components/agent/Mode/Chat/Panel/MessageList';
import { Welcome } from '@/components/agent/Mode/Chat/Panel/Welcome';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useChatScroll } from '@/hooks/agent/Mode/Chat/useChatScroll';
import { useFetchMessages } from '@/hooks/agent/Mode/Chat/useFetchMessages';

export function Panel() {
    const scrollAreaRef = useChatScroll();

    const messages = useFetchMessages();
    if (!messages) return null;

    return (
        <div className="flex flex-col flex-1 bg-gray-50 dark:bg-gray-800 min-h-0">
            <ScrollArea
                className="flex-1 min-h-0"
                ref={scrollAreaRef}
            >
                <div className="space-y-4 mx-auto p-4 pb-6 max-w-4xl">
                    {messages.length === 0
                        ? <Welcome />
                        : <MessageList messages={messages} />
                    }
                    <LoadingIndicator />
                </div>
            </ScrollArea>
        </div>
    );
}
