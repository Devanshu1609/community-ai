import { usePathname } from 'next/navigation';

import { Input } from '@/components/ui/input';
import { integrationModes } from '@/lib/constants/chat';
import { useAgentStore } from '@/lib/store/agent/agentStore';

export function MessageInput() {
  const { input, setInput, status } = useAgentStore();

  const pathname = usePathname();
  const currentMode = pathname.split('/')[1];
  const name = integrationModes.find((m) => m.id === currentMode)?.name;

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInput(e.target.value);
  };

  return (
    <div className="relative flex-1">
      <Input
        value={input}
        onChange={handleInputChange}
        placeholder={`Ask ${name || "the assistant"}...`}
        disabled={status !== "ready"}
        autoFocus
      />
    </div>
  );
}
