import { usePathname } from 'next/navigation';

import { useToast } from '@/hooks/use-toast';
import { useAgentStore } from '@/lib/store/agent/agentStore';

import { MessageInput } from './InputBox/MessageInput';
import { SubmitButton } from './InputBox/SubmitButton';

export function InputBox() {
  const pathname = usePathname();
  const mode = pathname.split('/')[1];
  const chatId = pathname.split('/')[2];

  const { sendMessage } = useAgentStore();
  const { toast } = useToast();

  const handleSubmit = async (event?: React.FormEvent) => {
    if (event) event.preventDefault();
    try {
      await sendMessage(chatId, mode);
    } catch (err: any) {
      toast({
        variant: 'destructive',
        title: 'Error sending message',
        description: err.message || 'Could not send your message.',
      });
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 p-4 dark:border-gray-700 border-t">
      <div className="mx-auto max-w-4xl">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <MessageInput />
          <SubmitButton />
        </form>
      </div>
    </div>
  );
}