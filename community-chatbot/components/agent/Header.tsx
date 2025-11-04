import { HeaderActions } from '@/components/agent/Header/HeaderActions';
import { HeaderLogo } from '@/components/agent/Header/HeaderLogo';
import { HeaderStatus } from '@/components/agent/Header/HeaderStatus';

export function ChatHeader() {
    return (
        <div className="flex items-center gap-3 bg-white dark:bg-gray-800 p-4 dark:border-gray-700 border-b h-20">
            <HeaderLogo />
            <HeaderStatus />
            <HeaderActions />
        </div>
    );
}