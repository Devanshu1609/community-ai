import { Bot, User } from 'lucide-react';

import { Avatar, AvatarFallback } from '@/components/ui/avatar';

interface ChatAvatarProps {
  role: "user" | "assistant"
}

export function ChatAvatar({ role }: ChatAvatarProps) {
  if (role === "user") {
    return (
      <Avatar className="w-8 h-8">
        <AvatarFallback className="bg-gray-600 text-white">
          <User className="w-4 h-4" />
        </AvatarFallback>
      </Avatar>
    )
  } else {
    return (
      <Avatar className="w-8 h-8">
        <AvatarFallback className="bg-blue-600 text-white">
          <Bot className="w-4 h-4" />
        </AvatarFallback>
      </Avatar>
    )
  }
}
