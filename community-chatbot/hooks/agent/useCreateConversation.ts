import { usePathname, useRouter } from 'next/navigation';

import { useToast } from '@/hooks/use-toast';
import { useAgentStore } from '@/lib/store/agent/agentStore';

export function useCreateConversation() {
	const router = useRouter();
	const { toast } = useToast();
	const pathname = usePathname();
	const currentMode = pathname.split('/')[1];
	const { createConversation } = useAgentStore();

	const handleCreateConversation = async () => {
		try {
			const newId = await createConversation(currentMode);
			toast({ title: "New chat created" });
			router.push(`/${currentMode}/${newId}`);
		} catch (error) {
			const errorMessage = error instanceof Error ? error.message : "Please try again.";
			toast({
				variant: "destructive",
				title: "Error creating chat",
				description: errorMessage
			});
		}
	};

	return { handleCreateConversation };
}
