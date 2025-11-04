import Image from 'next/image';
import { usePathname } from 'next/navigation';

import { Button } from '@/components/ui/button';
import { integrationModes, quickActions } from '@/lib/constants/chat';
import { useAgentStore } from '@/lib/store/agent/agentStore';

export function Welcome() {
  const pathname = usePathname();
  const currentMode = pathname.split('/')[1];
  const chatId = pathname.split("/")[2];
  const { handleQuickAction, status } = useAgentStore();

  const currentModeDetail = integrationModes.find((m) => m.id === currentMode);
  if (!currentModeDetail) return;

  return (
    <div className="py-8 text-center">
      <div className="bg-white dark:bg-gray-900 shadow-sm p-6 rounded-lg">
        <div className="flex flex-col justify-center items-center mb-4">
          {currentModeDetail &&
            <Image
              src={currentModeDetail.image}
              alt={currentModeDetail.name}
              width={32}
              height={32} />
          }
          <div className="flex flex-col justify-center items-center w-full">
            <h3 className="font-semibold text-lg">{currentModeDetail.name}</h3>
            <p className="text-gray-600 dark:text-gray-400 text-sm">{currentModeDetail.description}</p>
          </div>
        </div>
        <p className="mb-4 text-gray-600 dark:text-gray-400">
          Hello! I'm your Mifos Community assistant. I can help with general questions or connect to Slack,
          Jira, and GitHub. Select a currentMode above to get started!
        </p>
        <div>
          <p className="mb-3 text-gray-500 text-sm">Quick actions:</p>
          <div className="gap-2 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
            {quickActions.map((action, index) => (
              <Button
                key={index}
                variant="outline"
                size="sm"
                onClick={() => handleQuickAction(action, chatId, currentMode)}
                disabled={status === "error"}
                className="hover:bg-blue-50 dark:hover:bg-gray-700 hover:border-blue-300 text-xs"
              >
                {action}
              </Button>

            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
