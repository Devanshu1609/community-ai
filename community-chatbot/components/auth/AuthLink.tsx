import Link from 'next/link';

export function AuthLink({ href, children }: { href: string; children: React.ReactNode }) {
	return (
		<div className="text-sm text-center">

			<Link href={href} className="text-blue-600 hover:underline">
				{children}
			</Link>
		</div>
	)
}