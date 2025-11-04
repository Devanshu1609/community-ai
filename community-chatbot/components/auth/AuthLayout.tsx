"use client"

import Image from 'next/image';

import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from '@/components/ui/card';
import { LayoutProps } from '@/types/auth/types';

export function AuthLayout({
	title,
	description,
	children,
}: LayoutProps) {
	return (
		<div className="flex justify-center items-center bg-gradient-to-br from-blue-50 to-indigo-100 p-4 w-full min-h-screen">
			<Card className="w-full max-w-md">
				<CardHeader className="space-y-1 text-center">
					<div className="flex justify-center items-center mb-4">
						<div className="p-3 rounded-full size-20">
							<Image
								src="/mifos.png"
								alt="logo"
								width={200}
								height={200}
							/>
						</div>
					</div>
					<CardTitle className="font-bold text-2xl">
						{title}
					</CardTitle>
					<CardDescription>
						{description}
					</CardDescription>
				</CardHeader>
				<CardContent className="space-y-4">
					{children}
				</CardContent>
			</Card>
		</div>
	)
}