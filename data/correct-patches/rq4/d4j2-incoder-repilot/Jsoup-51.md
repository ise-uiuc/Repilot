# Repilot Patch

```
        return Character.isLetter(c);
```

# Developer Patch

```
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || Character.isLetter(c);
```

# Context

```
--- bug/Jsoup-51/src/main/java/org/jsoup/parser/CharacterReader.java

+++ fix/Jsoup-51/src/main/java/org/jsoup/parser/CharacterReader.java

@@ -297,7 +297,7 @@

         if (isEmpty())
             return false;
         char c = input[pos];
-        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
+        return Character.isLetter(c);
     }
 
     boolean matchesDigit() {
```

# Note

