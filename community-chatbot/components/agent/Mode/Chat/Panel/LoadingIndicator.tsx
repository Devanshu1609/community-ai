import { Loader2 } from 'lucide-react';

import { useAgentStore } from '@/lib/store/agent/agentStore';

export function LoadingIndicator() {
  const { status } = useAgentStore();

  if (status !== "submitted" && status !== "streaming")
    return null;

  return (
    <div className="flex justify-start gap-3">
      <div className="bg-white dark:bg-gray-900 shadow-sm px-4 py-3 border rounded-lg">
        <div className="flex items-center gap-2">
          <Loader2 className="w-4 h-4 animate-spin" />
          <span className="text-gray-600 dark:text-gray-400 text-sm">
            {status === "submitted"
              ? "Thinking..."
              : "Typing..."
            }
          </span>
        </div>
      </div>
    </div>
  );
}
