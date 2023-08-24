# Repilot Patch

```
if (element == search)
```

# Developer Patch

```
            if (element == search)
```

# Context

```
--- bug/Jsoup-43/src/main/java/org/jsoup/nodes/Element.java

+++ fix/Jsoup-43/src/main/java/org/jsoup/nodes/Element.java

@@ -571,7 +571,7 @@

 
         for (int i = 0; i < elements.size(); i++) {
             E element = elements.get(i);
-            if (element.equals(search))
+if (element == search)
                 return i;
         }
         return null;
```

# Note

