import { getAuth, onAuthStateChanged, User } from 'firebase/auth';
import { StateCreator } from 'zustand';

export interface authSlice {
    currentUser: User | null;

    initAuth: () => () => void;
}

export const createAuthSlice: StateCreator<
    authSlice,
    [],
    [],
    authSlice
> = (set) => ({
    currentUser: null,

    initAuth: () => {
        const auth = getAuth();
        const unsubscribe = onAuthStateChanged(auth, (user) => {
            set({ currentUser: user });
        });
        return unsubscribe;
    }
});
