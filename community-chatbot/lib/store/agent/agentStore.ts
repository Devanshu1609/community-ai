import type { StoreApi, UseBoundStore } from 'zustand';
import { create } from 'zustand';

import { authSlice, createAuthSlice } from '@/lib/store/agent/slices/authSlice';
import {
  conversationSlice, createConversationSlice,
} from '@/lib/store/agent/slices/conversationSlice';

import { createMessageSlice, messageSlice } from './slices/messageSlice';

type AgentStore = authSlice & conversationSlice & messageSlice;

export const useAgentStore: UseBoundStore<StoreApi<AgentStore>> =
  create<AgentStore>((set, get, api) => ({
    // Spread slices into the root state. Pass (set, get, api) so slices can use all three.
    ...createAuthSlice(set, get, api),
    ...createConversationSlice(set, get, api),
    ...createMessageSlice(set, get, api),
  }));
