"use client"

import { useState } from 'react';

import { GoogleAuthProvider, signInWithPopup } from 'firebase/auth';
import { Loader2 } from 'lucide-react';
import { useRouter } from 'next/navigation';

import { auth } from '@/app/firebase/config';
import { GoogleIcon } from '@/components/icons';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';

export function GoogleSignInButton() {
	const [loading, setLoading] = useState(false)
	const router = useRouter()
	const { toast } = useToast()

	const handleGoogleSignIn = async () => {
		setLoading(true)
		try {
			const provider = new GoogleAuthProvider()
			await signInWithPopup(auth, provider)
			router.push("/")
			toast({
				title: "Welcome back!",
				description: "Signed in successfully."
			})
		} catch {
			toast({
				variant: "destructive",
				title: "Google sign in failed",
				description: "Something went wrong.",
			})
		} finally {
			setLoading(false)
		}
	}

	return (
		<Button
			variant="outline"
			className="bg-transparent w-full"
			onClick={handleGoogleSignIn}
			disabled={loading}
		>
			{loading ? <Loader2 className="mr-2 w-4 h-4 animate-spin" /> : <GoogleIcon />}
			Continue with Google
		</Button>
	)
}