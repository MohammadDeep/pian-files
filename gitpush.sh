#!/bin/bash

# گرفتن پیام کامیت از ورودی یا استفاده از پیش‌فرض
commit_msg=${1:-"🚀 quick commit"}

# بررسی اینکه داخل یک ریپوی گیت هستیم
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "❌ این مسیر یک گیت ریپازیتوری نیست."
    exit 1
fi

# گرفتن نام برنچ فعلی
branch=$(git branch --show-current)

echo "📂 برنچ فعلی: $branch"

# دریافت تغییرات جدید از ریموت قبل از push
echo "🔄 Pulling remote changes before pushing..."
git pull origin "$branch" --rebase

# افزودن تمام فایل‌ها
echo "➕ افزودن فایل‌ها..."
git add .

# کامیت اگر چیزی برای کامیت هست
echo "📝 Commiting: $commit_msg"
if git diff --cached --quiet; then
    echo "ℹ️ چیزی برای commit وجود ندارد."
else
    git commit -m "$commit_msg"
fi

# پوش به ریموت
echo "🚀 Pushing to origin/$branch"
git push -u origin "$branch"

