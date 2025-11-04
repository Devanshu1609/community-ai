import { Separator } from '@/components/ui/separator';

export function OrSeparator(
	{ text }: { text: string }
) {
	return (
		<div className="relative">
			<Separator className="w-full" />
			<div className="absolute inset-0 flex justify-center items-center">
				<span className="bg-background px-2 text-muted-foreground text-xs uppercase">
					{text}
				</span>
			</div>
		</div>
	)
}