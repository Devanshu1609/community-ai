import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

import { ChatAvatar } from '@/components/agent/Mode/Chat/Panel/ChatAvatar';
import { Message } from '@/types/chat/types';

interface MessageListProps {
  messages:Message[]
}

export function MessageList({ messages }: MessageListProps) {
  return (
    <>
      {messages.map((message) => (
        <div key={message.id} className={`flex gap-3 ${message.role === "user" ? "justify-end" : "justify-start"}`}>
          {message.role === "assistant" && <ChatAvatar role="assistant" />}
          <div
            className={`max-w-[80%] rounded-lg px-4 py-3 ${message.role === "user"
              ? "bg-blue-600 text-white ml-auto"
              : "bg-white dark:bg-gray-700 shadow-sm border dark:border-gray-600"
              }`}
          >
            <div className="whitespace-pre-wrap">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
            </div>
          </div>
          {message.role === "user" && <ChatAvatar role="user" />}
        </div>
      ))}
    </>
  );
}
