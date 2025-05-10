import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

const eslintConfig = [
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  {
    files: ["**/*.ts", "**/*.tsx"], // Apply these rules to TypeScript files
    rules: {
      "@typescript-eslint/no-explicit-any": "off", // Turn off the `no-explicit-any` rule
      "@typescript-eslint/no-empty-interface": "off", // Turn off the `typescript/no-empty-object` rule
      "@typescript-eslint/no-unused-vars": [
        "warn",
        {
          argsIgnorePattern: "^_", // Ignore unused arguments that start with `_`
          varsIgnorePattern: "^_", // Ignore unused variables that start with `_`
        },
      ],
    },
  },
];

export default eslintConfig;
