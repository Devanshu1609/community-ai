import { useEffect, useState } from 'react';

import { onAuthStateChanged } from 'firebase/auth';
import { useRouter } from 'next/navigation';

import { auth } from '@/app/firebase/config';

export function useAuthRedirect() {
	const [loading, setLoading] = useState(true)
	const router = useRouter()

	useEffect(() => {
		const unsubscribe = onAuthStateChanged(auth, (user) => {
			if (user) {
				router.replace("/")
			} else {
				setLoading(false)
			}
		})

		return () => unsubscribe()
	}, [router])

	return { loading }
}