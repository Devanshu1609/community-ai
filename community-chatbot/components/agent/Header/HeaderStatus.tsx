"use client";

import { usePathname } from 'next/navigation';

import { Badge } from '@/components/ui/badge';
import { integrationModes } from '@/lib/constants/chat';

export function HeaderStatus() {
    const pathname = usePathname();
    const currentMode = pathname.split('/')[1];
    const name = integrationModes.find((m) => m.id === currentMode)?.name;

    return (
        <div className="flex-1">
            <h1 className="block font-bold text-sm sm:text-xl truncate">
                Mifos Assistant
            </h1>
            <div className="hidden sm:flex items-center gap-2">
                <div className="bg-green-500 rounded-full w-2 h-2"></div>
                <span className="text-green-600 text-xs sm:text-sm">Online</span>
                {name && (
                    <Badge variant="outline" className="text-xs">
                        {name}
                    </Badge>
                )}
            </div>
        </div>
    );
}